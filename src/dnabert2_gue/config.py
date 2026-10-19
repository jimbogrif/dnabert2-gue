"""Paths, model constants and the dataset manifest for the DNABERT-2 layer x
pooling x model sweep on GUE.

Model/training definitions across this package are lifted verbatim from the
original DNABERT-2 simple-GUE benchmark notebook. The only deliberate changes
are three documented fixes; see ``model`` (fixes 1 and 2) and ``linear``
(fix 3).

Every path can be overridden by an environment variable; the defaults are
repo-relative (``data/GUE``, ``layer_embed``, ``results``, ``figures``).
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths and dataset manifest
# --------------------------------------------------------------------------

# src/dnabert2_gue/config.py -> parents[2] is the repo root. Asserted, because
# a wrong answer here would write 53 GB of embeddings to the wrong place.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert (PROJECT_ROOT / "pyproject.toml").exists(), (
    f"PROJECT_ROOT resolved to {PROJECT_ROOT}, which has no pyproject.toml; "
    "install with `pip install -e .` from a clone of the repo")
ROOT = str(PROJECT_ROOT)
GUE_ROOT = os.environ.get("DNABERT2_GUE_ROOT", str(PROJECT_ROOT / "data" / "GUE"))
EMBED_DIR = os.environ.get("DNABERT2_EMBED_DIR", str(PROJECT_ROOT / "layer_embed"))
RESULTS_DIR = os.environ.get("DNABERT2_RESULTS_DIR", str(PROJECT_ROOT / "results"))
FIGURES_DIR = os.environ.get("DNABERT2_FIGURES_DIR", str(PROJECT_ROOT / "figures"))

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
N_LAYERS = 12          # output_all_encoded_layers returns 12 encoder layers
FINAL_LAYER = 12
HIDDEN = 768
POOLS = ("mean", "cls")
SPLITS = ("train", "dev", "test")

# The 26 GUE datasets in scope: the full 28-dataset GUE benchmark minus
# splice/reconstructed (SSP) and virus/covid, both excluded by the user.
# GUE+ (EPI, fungi, virus/species_40) is out of scope entirely.
TASKS = {
    "EMP": ("EMP", ["H3", "H3K14ac", "H3K36me3", "H3K4me1", "H3K4me2",
                    "H3K4me3", "H3K79me3", "H3K9ac", "H4", "H4ac"]),
    "TF-M": ("mouse", ["0", "1", "2", "3", "4"]),
    "TF-H": ("tf", ["0", "1", "2", "3", "4"]),
    "PD":   ("prom", ["prom_300_all", "prom_300_notata", "prom_300_tata"]),
    "CPD":  ("prom", ["prom_core_all", "prom_core_notata", "prom_core_tata"]),
}

# Splice-site prediction is deliberately NOT in TASKS: it is 3-class where all
# 26 above are binary, so folding it in would silently change what every
# aggregate over manifest() means. It is declared separately and reached only
# by the positional-readout arm, which asks for it by name.
EXTRA_TASKS = {
    "SSP": ("splice", ["reconstructed"]),
}


def manifest(include_extra=False):
    """[(task, dataset_key, csv_dir), ...] for the 26 in-scope datasets.

    ``dataset_key`` matches the original notebook's "<category>__<dataset>"
    convention so filenames stay comparable with its cached embeddings.

    ``include_extra=True`` appends EXTRA_TASKS (currently SSP). It defaults to
    False because every existing arm iterates this function to enumerate its
    work, and those arms are defined over the 26 binary datasets.
    """
    rows = []
    tasks = {**TASKS, **EXTRA_TASKS} if include_extra else TASKS
    for task, (cat, names) in tasks.items():
        for n in names:
            rows.append((task, f"{cat}__{n}", os.path.join(GUE_ROOT, cat, n)))
    return rows


def dataset_dir(dataset):
    """CSV directory for a dataset key, in-scope or extra.

    Call sites that need a path for a single named dataset should use this
    rather than indexing ``manifest()``, so they keep working for SSP.
    """
    paths = {key: p for _, key, p in manifest(include_extra=True)}
    if dataset not in paths:
        raise KeyError(f"unknown dataset {dataset!r}; "
                       f"known: {sorted(paths)}")
    return paths[dataset]


# Lookup only -- includes the extra datasets so task labelling works for them.
# Nothing should iterate this to enumerate the benchmark; use manifest().
DATASET_TASK = {key: task for task, key, _ in manifest(include_extra=True)}


# --------------------------------------------------------------------------
# LoRA fine-tuning arm
# --------------------------------------------------------------------------

LORA_SEEDS = (42, 43, 44)
LORA_BATCH = 128             # bf16; ~817 seq/s, 6.4 GiB peak on an RTX 4080
LORA_MAX_EPOCHS = 40         # a backstop, not a budget: patience does the stopping
LORA_PATIENCE = 5            # epochs without dev improvement before stopping
LORA_WARMUP_RATIO = 0.06
LORA_WEIGHT_DECAY = 0.01
LORA_R = 8                   # r and alpha fixed: alpha/r = 2.0 throughout
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MAX_LENGTH = 512
LORA_CALIB_LRS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
LORA_CALIB_DATASETS = ("mouse__3", "prom__prom_core_tata", "EMP__H3K4me3")

LORA_VARIANTS = {
    "attn":     ["Wqkv"],
    "attn_mlp": ["Wqkv", "mlp.gated_layers", "mlp.wo"],
}


# --------------------------------------------------------------------------
# Positional-readout arm
# --------------------------------------------------------------------------
#
# Asks whether whole-sequence mean pooling discards positional information
# that a frozen DNABERT-2 embedding actually carries. Three datasets spanning
# the measured range of how positional a GUE task is -- share of per-position
# information content falling in the top 20 positions, over the uniform
# expectation 20/L for that sequence length:
#
#   splice__reconstructed  400 bp   15.5x   (donor/acceptor at a fixed index)
#   prom__prom_300_all     300 bp    2.8x   (weak anchor near the 3' end)
#   tf__0                  101 bp    1.1x   (no positional anchor)
#
# Windows are a UNIFORM tiling, identical rule for all three datasets. An
# anchor on splice's known junction would score better but is a task-informed
# choice the controls cannot have, which is exactly the comparison at issue.

POSITIONAL_DATASETS = ("splice__reconstructed", "prom__prom_300_all", "tf__0")
POSITIONAL_N_WINDOWS = 8     # 50 bp on splice, 37.5 on PD, 12.6 on tf__0
POSITIONAL_SHUFFLE_SEED = 0  # per-sequence window permutation for the control

# Stored readouts; the two derived ones are reconstructed from the win cache
# at probe time rather than written to disk (saves a second 41 GB pass).
POSITIONAL_STORED = ("mean", "cls", f"win{POSITIONAL_N_WINDOWS}")
POSITIONAL_DERIVED = (f"win{POSITIONAL_N_WINDOWS}_best",    # dev picks 1 window
                      f"win{POSITIONAL_N_WINDOWS}shuf")     # dimension control
POSITIONAL_READOUTS = POSITIONAL_STORED + POSITIONAL_DERIVED
