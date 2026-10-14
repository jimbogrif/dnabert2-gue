"""DNABERT-2 frozen-embedding layer x pooling x classifier sweep on GUE."""

from .config import (TASKS, POOLS, SPLITS, N_LAYERS, FINAL_LAYER, HIDDEN,
                     MODEL_NAME, GUE_ROOT, EMBED_DIR, RESULTS_DIR, FIGURES_DIR,
                     manifest, DATASET_TASK)
from .model import load_model, embed_batch_layers
from .embed import embed_path, label_path, load_split
from .linear import grid_search_lr, CS, CLASS_WEIGHTS, CS_NOTEBOOK
from .mlp import run_mlp, prep_data, evaluate, train_model
from .config import (LORA_SEEDS, LORA_BATCH, LORA_MAX_EPOCHS, LORA_CALIB_LRS,
                     LORA_CALIB_DATASETS, LORA_VARIANTS)
from .lora import run_lora, build_lora_model
