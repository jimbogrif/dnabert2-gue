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


def manifest():
    """[(task, dataset_key, csv_dir), ...] for the 26 in-scope datasets.

    ``dataset_key`` matches the original notebook's "<category>__<dataset>"
    convention so filenames stay comparable with its cached embeddings.
    """
    rows = []
    for task, (cat, names) in TASKS.items():
        for n in names:
            rows.append((task, f"{cat}__{n}", os.path.join(GUE_ROOT, cat, n)))
    return rows


DATASET_TASK = {key: task for task, key, _ in manifest()}


# --------------------------------------------------------------------------
# LoRA fine-tuning arm
# --------------------------------------------------------------------------

LORA_SEEDS = (42, 43, 44)
LORA_BATCH = 128             # bf16; ~817 seq/s, 6.4 GiB peak on an RTX 4080
LORA_MAX_EPOCHS = 5
LORA_WARMUP_RATIO = 0.06
LORA_WEIGHT_DECAY = 0.01
LORA_R = 8                   # r and alpha fixed: alpha/r = 2.0 throughout
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_MAX_LENGTH = 512
LORA_CALIB_LRS = (1e-4, 3e-4, 1e-3)
LORA_CALIB_DATASETS = ("mouse__3", "prom__prom_core_tata", "EMP__H3K4me3")

LORA_VARIANTS = {
    "attn":     ["Wqkv"],
    "attn_mlp": ["Wqkv", "mlp.gated_layers", "mlp.wo"],
}
