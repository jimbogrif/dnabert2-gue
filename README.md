# DNABERT-2 on GUE: layer × pooling × classifier, LoRA, and a bag-of-tokens floor

This repository accompanies the blog post
**[DNABERT-2 classification](https://jimbogrif.github.io/posts/DNABERT2-classification/)**.
The post presents and discusses the results. This repo holds the code that
produced them and the result tables it wrote.

It runs four arms on 26 of the 28 GUE datasets, all under one protocol: dev
selects each configuration, and test is scored once.

| Arm | What is trained | `model_type` |
|---|---|---|
| Frozen probe, linear | logistic regression on cached DNABERT-2 embeddings, every layer × mean / `[CLS]` pooling | `LR` |
| Frozen probe, MLP | a 256-unit MLP on the same cached embeddings, at three selected layers per task | `MLP` |
| LoRA fine-tune | rank-8 adapters on attention (`Wqkv`) plus a trained pooler and head | `LoRA-attn` |
| Bag-of-tokens floor | the frozen arms' LR and MLP heads on k-mer (k = 3–6) or DNABERT-2 BPE-token frequencies, with no transformer | `kmer-LR`, `kmer-MLP`, `bpe-LR`, `bpe-MLP` |

## Repository layout

```
src/dnabert2_gue/       the installable package
  config.py             paths, dataset manifest, LoRA settings
  model.py              DNABERT-2 loading and all-layer embedding
  embed.py              cached-embedding paths and loading
  linear.py             logistic-regression grid (shared by the frozen and bag arms)
  mlp.py                TaperedMLP, training loop, one MLP run
  lora.py               LoRA model construction, training, one LoRA run
  bags.py               k-mer / BPE featurizers and the bag-arm heads
scripts/                numbered pipeline steps (see Reproducing)
results/                committed result tables (see Outputs)
figures/                committed figures (see Figures)
```

## Requirements

| | |
|---|---|
| GPU | CUDA card, ≥6 GB VRAM (measured: RTX 4080, 16 GB) |
| Disk | ~55 GB free: 53 GB embeddings + 1.1 GB GUE |
| CPU | 20 cores on the reference machine; `03_lr_sweep.py` scales with core count |
| First run | Downloads the DNABERT-2 checkpoint (~500 MB) from HuggingFace. `HF_HUB_OFFLINE=1` works on every later run. |
| Wall clock | verify ~2 min · embed ~10 min · LR sweep ~2.0 h · MLP sweep ~51 min · LoRA calibration ~67 min · LoRA full ~6.4 h · bag floor ~50 min · tables <1 min |

**Only `01_verify_layers.py`, `02_embed_all.py` and `05_lora_sweep.py` need a
GPU.** All three exit with a clear message if there isn't one, rather than
falling back to a multi-day CPU run. Once `layer_embed/` exists the frozen arms
run on CPU; the bag floor's MLP half uses a GPU if there is one. Every result is
committed under `results/`, so without a GPU you can still rebuild the tables.

## Reproducing

```bash
git clone <this repo> dnabert2-gue && cd dnabert2-gue
pip install -e .                       # editable install required (see below)
python scripts/00_get_gue.py           # prints download steps, verifies data/GUE
python scripts/smoke_test.py           # ~30 s on a GPU: whole stack on mouse__3

python scripts/01_verify_layers.py     # correctness gate (GPU)
python scripts/02_embed_all.py         # ~10 min, writes 53 GB (GPU)
OMP_NUM_THREADS=3 python scripts/03_lr_sweep.py   # ~2.0 h (CPU)
python scripts/04_mlp_sweep.py 32      # ~51 min
python scripts/05_lora_sweep.py --mode calibrate             # ~67 min (GPU); stops at the decision gate
python scripts/05_lora_sweep.py --mode full --lr 1e-3 --variant attn   # ~6.4 h (GPU)
python scripts/07_bag_baseline.py      # ~50 min, LR half on CPU, MLP half on GPU if present
python scripts/06_build_tables.py      # <1 min, rebuilds the derived tables
```

| Script | Does |
|---|---|
| `00_get_gue.py` | Prints GUE download steps and checks that all 26 datasets are present in `data/GUE` |
| `smoke_test.py` | Embeds `mouse__3` into a temp dir and runs one LR cell, to check the install end to end |
| `01_verify_layers.py` | Checks that the all-layer embedding path matches the model's own forward pass bit for bit; also audits token lengths and `[SEP]` placement |
| `02_embed_all.py` | Writes mean- and `[CLS]`-pooled embeddings for all 12 layers to `layer_embed/` |
| `03_lr_sweep.py` | Logistic-regression grid over every dataset × layer × pooling |
| `04_mlp_sweep.py` | MLP at the three layers per task in `results/selected_layers.csv`, 3 seeds |
| `05_lora_sweep.py` | `--mode calibrate` picks the learning rate on three datasets and checks a decision gate; `--mode full` runs all 26 datasets × 3 seeds |
| `06_build_tables.py` | Rebuilds the derived tables from the input CSVs |
| `07_bag_baseline.py` | Bag-of-tokens floor: every featurization × LR grid and MLP |

`06_build_tables.py` reads whichever LoRA and bag outputs exist, so you can run
it at any point. `requirements.lock.txt` pins the full dependency closure at
the versions that produced `results/`. For an exact environment, run
`pip install -r requirements.lock.txt` before `pip install -e .`. The install
must be editable: the package resolves its default paths from the repo root and
refuses to run from a non-editable install.

`02_embed_all.py`, `03_lr_sweep.py`, `04_mlp_sweep.py`, `05_lora_sweep.py` and
`07_bag_baseline.py` all resume from partial state. `03_lr_sweep.py` and
`07_bag_baseline.py` run `cpu_count // 3` parallel jobs by default (6 jobs ×
`OMP_NUM_THREADS=3` measured best on 20 cores); set `SWEEP_N_JOBS` to override.
Timings were measured on an RTX 4080 with 20 CPU cores.

### Paths

With the default layout nothing needs to be set: GUE goes in `data/GUE` and
embeddings in `layer_embed/`, both gitignored. The variables are an escape
hatch for data that must live elsewhere (cluster scratch, another volume).

| Variable | Default | Meaning |
|---|---|---|
| `DNABERT2_GUE_ROOT` | `data/GUE` | GUE benchmark root (contains `EMP/`, `mouse/`, `prom/`, `tf/`) |
| `DNABERT2_EMBED_DIR` | `layer_embed` | cached per-layer embeddings (53 GB) |
| `DNABERT2_RESULTS_DIR` | `results` | sweep outputs and derived tables |
| `DNABERT2_FIGURES_DIR` | `figures` | figure output |

## Figures

| File | Shows | Written by |
|---|---|---|
| `figures/layer_profile.png` | dev MCC vs. encoder layer, per task | no script (see TODO) |
| `figures/lr_vs_mlp.png` | paired LR vs. MLP test MCC, per cell | no script (see TODO) |
| `figures/vs_paper.png` | best frozen config vs. paper Table 6, per dataset | no script (see TODO) |
| `figures/lora_calibration.png` | LoRA dev MCC vs. learning rate, against the best frozen dev MCC | `05_lora_sweep.py --mode calibrate` |

## Outputs

**Input** files are primary evidence from the sweeps. **Derived** files are
rebuilt from the inputs by `06_build_tables.py`. **Record** files document a
check or a one-off run.

| File | Kind | Contents |
|---|---|---|
| `results/lr_layer_sweep.csv` | input | the complete 12-layer LR grid (11,232 rows) |
| `results/mlp_results.csv` | input | per-seed MLP runs (468 rows) |
| `results/lora_results.csv` | input | per-seed LoRA runs, 26 datasets × 3 seeds (78 rows) |
| `results/bag_results.csv` | input | per-dataset selected bag rows for every featurization: LR, and MLP per seed (520 rows) |
| `results/selected_layers.csv` | input | the three layers per task fed to the MLP arm |
| `results/paper_table6_dnabert2.csv` | input | the paper's fine-tuned DNABERT-2 MCC (Table 6) |
| `results/history_table.csv` | derived | every arm in one long table |
| `results/history_table_aggregated.csv` | derived | seed-aggregated, best per cell |
| `results/best_frozen_vs_paper.csv` | derived | best dev-selected frozen config per dataset vs. Table 6 |
| `results/best_vs_paper_with_finetune.csv` | derived | best frozen and best LoRA side by side vs. Table 6 |
| `results/bag_best_per_dataset.csv` | derived | best bag config per dataset for each bag `model_type` |
| `results/lora_calibration.csv` | record | the learning-rate calibration runs, with `at_grid_edge` |
| `results/bag_lr_grid.csv` | record | every bag LR grid point |
| `results/embed_manifest.csv` | record | per-split sequence counts, token lengths and embedding throughput |
| `results/layer_profile_task.csv` | record | per-task mean dev MCC by layer |
| `results/mlp_batch_benchmark.csv`, `mlp_batch_projection.json` | record | MLP batch-size benchmark and projected sweep cost |
| `results/verify_layers.json` | record | output of `01_verify_layers.py` |
| `results/epi_sep_check.txt` | record | `[SEP]` placement for a sequence pair |
| `layer_embed/` | local | 1,950 `.npy` files, 53 GB; regenerable in ~10 min; not committed |

The sweep scripts also write per-epoch training curves
(`results/*_histories.csv.gz`) and per-cell resume state (`results/lr_cells/`,
`results/bag_lr_cells/`). These stay local and are not committed.

### History table columns

`model_type`, `task`, `dataset`, `pool`, `layer`, `layer_role`, `seed`, `C`,
`class_weight`, `dev_mcc`, `test_mcc`, `dev_selected`, `best_epoch`,
`epochs_run`, `early_stopped`, `batch_size`, `hidden_dims`, `dropout`, `lr`,
`weight_decay`, `optimizer`, `max_epochs`, `patience`, `n_iter`,
`at_grid_edge`, `in_notebook_grid`, `hyperparams`, `trainable_params`,
`pct_trainable`, `feat`, `n_features`.

- **LR** rows cover the full 18-point grid at each selected layer; `test_mcc` is
  filled only on the dev-selected row. **MLP** rows are one per seed.
- **LoRA** rows use placeholder values, because fine-tuning has no readout layer
  and no free pooling choice: `layer = 0` (real encoder layers are 1–12),
  `layer_role = "finetuned"`, `pool = "cls-trained"`. LoRA settings go in
  `hyperparams`. `trainable_params` and `pct_trainable` are filled only for
  LoRA.
- **Bag** rows use `layer = -1`, `layer_role = "bag"`, `pool = "bag"`, with the
  featurization in `feat` and its width in `n_features`.
- `class_weight=None` is stored as the literal string `"none"`, so `pd.read_csv`
  does not read it as NaN.

## TODO

- **Confirm the blog post URL** at the top of this file.
- **Fill in the clone URL** in Reproducing (`<this repo>`).
- **Add a script for the three frozen-sweep figures.** No script in the repo
  produces `layer_profile.png`, `lr_vs_mlp.png` or `vs_paper.png`, and their
  source isn't in the repo.
- **Bring the figures up to date with the later arms.** `vs_paper.png` shows the
  frozen arm only; there is no figure for the LoRA or bag-of-tokens results.
- **Run the `attn_mlp` LoRA variant, or remove it.** `05_lora_sweep.py
  --variant attn_mlp` (adapters on the MLP blocks as well as attention) is
  implemented but has never been run, and every LoRA result is `attn`.
- **Reconcile the two token-length records.** `01_verify_layers.py` samples 200
  training sequences per dataset (`verify_layers.json`: max 115 tokens), while
  `embed_manifest.csv` covers every sequence in every split (max 116).
