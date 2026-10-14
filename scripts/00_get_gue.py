"""Fetch-and-verify for the GUE benchmark data.

GUE (~1.1 GB) cannot be committed and is distributed through Google Drive,
which has no stable direct-download URL, so this script does not download it.
It prints the manual steps, then verifies that all 26 in-scope datasets are
present under GUE_ROOT with train/dev/test.csv, so a partial download fails
here rather than 40 minutes into embedding. Exits non-zero on any gap.

Usage: 00_get_gue.py
"""

import os
import sys

import dnabert2_gue as SL

URL = "https://drive.google.com/file/d/1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-/view?usp=sharing"
SOURCE = "https://github.com/MAGICS-LAB/DNABERT_2#21-gue-genome-understanding-evaluation"
DEFAULT = os.path.join(SL.config.PROJECT_ROOT, "data", "GUE")

print(f"""GUE is not downloaded automatically. To fetch it:

  1. Download the archive linked from the DNABERT-2 README ({SOURCE}):
       {URL}
  2. Unpack it so the dataset folders (EMP/, mouse/, prom/, tf/, ...) sit at:
       {DEFAULT}
  3. Or, to keep the data elsewhere (cluster scratch, another volume):
       export DNABERT2_GUE_ROOT=/path/to/GUE

Checking GUE_ROOT = {SL.GUE_ROOT}
""", flush=True)

missing, per_task = [], {}
for task, dataset, path in SL.manifest():
    have = [s for s in SL.SPLITS if os.path.isfile(os.path.join(path, f"{s}.csv"))]
    per_task.setdefault(task, [0, 0])
    per_task[task][1] += 1
    if len(have) == len(SL.SPLITS):
        per_task[task][0] += 1
    else:
        missing.append((dataset, sorted(set(SL.SPLITS) - set(have))))

for task, (ok, n) in per_task.items():
    print(f"  {task:5s} {ok:2d}/{n} datasets complete", flush=True)
n_ok = sum(ok for ok, _ in per_task.values())
print(f"  total {n_ok}/{len(SL.manifest())}", flush=True)

if missing:
    for dataset, splits in missing:
        print(f"  MISSING {dataset}: {', '.join(splits)}", flush=True)
    sys.exit(1)
print("GUE OK", flush=True)
