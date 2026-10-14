# DNABERT-2 frozen-embedding sweep: layer × pooling × classifier on GUE

Extends a simple DNABERT-2 GUE benchmark notebook from a single
final-layer readout to a full sweep over all 12 encoder layers, both pooling
strategies, and two classifier heads, on 26 of the 28 GUE datasets.

Nothing here is fine-tuned. DNABERT-2's weights are frozen throughout; only the
classifier on top of the cached embeddings is trained. (A LoRA fine-tuning arm
is planned; `dnabert2_gue.lora` and `scripts/05_lora_sweep.py` are stubs that
fix its interface.)

## Requirements

| | |
|---|---|
| GPU | CUDA card, ≥6 GB VRAM (measured: RTX 4080, 16 GB) |
| Disk | ~55 GB free — 53 GB embeddings + 1.1 GB GUE |
| CPU | 20 cores on the reference machine; `03_lr_sweep.py` scales with core count |
| First run | Downloads the DNABERT-2 checkpoint (~500 MB) from HuggingFace. `HF_HUB_OFFLINE=1` works on every subsequent run. |
| Wall clock | verify ~2 min · embed ~10 min · LR sweep ~2.0 h · MLP sweep ~51 min · tables <1 min |

**Only `01_verify_layers.py` and `02_embed_all.py` need the GPU**, and both exit
with a clear message if none is found rather than falling back to a multi-day
CPU run. Once `layer_embed/` exists everything downstream is CPU-only, and every
result is committed under `results/`, so without a card you can still rerun the
LR and MLP arms from existing embeddings, or skip straight to the tables.

---

## Headline

**Where you read the model out matters far more than what you put on top of it.**
Median effect on test MCC, measured across the sweep:

| Choice | Median Δ test MCC | Max Δ | Consistency |
|---|---|---|---|
| Best-of-top-2 layer vs. the final layer | **+0.050** | +0.320 | helps in 102 / 104 |
| Mean pooling vs. `[CLS]` | **+0.039** | — | helps in 142 / 156 |
| 256-unit MLP vs. logistic regression | **−0.002** | +0.052 | helps in 67 / 156 |

The two choices that cost nothing — which layer to read, how to pool — are worth
roughly 25× the median gain from replacing logistic regression with a neural
head. The original notebook used the final layer, which is the single worst of
the three defaults.

Frozen embeddings with the best configuration reach **88.7 % of the paper's
fine-tuned score** on average (range 74.6–104.9 %), with no gradient ever
flowing into DNABERT-2.

## Figures

- `figures/layer_profile.png` — dev MCC vs. encoder layer, per task
- `figures/lr_vs_mlp.png` — paired LR vs. MLP test MCC over 156 cells
- `figures/vs_paper.png` — best frozen config vs. paper Table 6, 26 datasets

## Result 1 — the final layer is never the right layer

Mean dev MCC over each task's datasets and both poolings; rank is out of 12.

| Task | Best layers | Layer-12 rank | Best dev MCC | Layer-12 dev MCC |
|---|---|---|---|---|
| CPD | 2, 5 | 11th | 0.600 | 0.550 |
| PD | 6, 5 | 10th | 0.786 | 0.709 |
| TF-H | 5, 4 | 11th | 0.530 | 0.468 |
| TF-M | 5, 3 | 12th | 0.641 | 0.468 |
| EMP | 6, 11 | 8th | 0.470 | 0.448 |

Peak probing accuracy sits at layers 2–6 for four of five tasks. TF-M is the
extreme: layer 5 gives dev MCC 0.641 against the final layer's 0.468, a 37 %
relative gain from readout depth alone. EMP is the only task with a near-flat
profile, and also the hardest of the five.

This reproduces, on DNABERT-2, the Nucleotide Transformer paper's observation
that the best probing layer is never the last one.

## Result 2 — the MLP head buys nothing

Across 156 (dataset × pooling × layer) cells, trained on exactly the same
cached features as the LR arm, the MLP's median advantage is −0.002 test MCC and
it wins 67 of 156 cells. Per task it is slightly negative on TF-H (−0.015 median)
and TF-M (−0.013), slightly positive on EMP (+0.002).

Taking the best configuration per dataset, the MLP is chosen 16 times and LR 10 —
but the margins are inside seed noise, and the MLP costs three seeds of GPU
training per cell against a one-second CPU fit.

The honest reading: with frozen embeddings and a linearly-decodable target, a
non-linear head has nothing left to extract. The representation, not the
classifier, is the bottleneck.

## Result 3 — pooling

Mean pooling beats `[CLS]` in 142 of 156 cells (median +0.039), and the gap
widens with depth. Of the 26 best-per-dataset configurations, 24 use mean
pooling. This is expected: DNABERT-2's checkpoint contains no trained pooler and
its MosaicBERT pre-training is MLM-only, so no loss term ever asks position 0 to
summarize the sequence.

## Scope

26 of the 28 GUE datasets: Yeast EMP (10), Mouse TF-M (5), Human TF-H (5),
Human PD (3), Human CPD (3). Excluded by request: splice/reconstructed (SSP,
position-specific and a poor fit for mean pooling) and virus/covid. GUE+ (EPI,
fungi, virus/species_40) is out of scope — those sequences are 5–10 kb and
would be truncated at 512 tokens.

All 26 are binary, 2-column (`sequence`, `label`), 70–500 bp, and sit at
0.44–0.59 positive fraction. The longest sequence observed in a 200-sequence
sample from each of the 26 training splits tokenizes to 115 tokens (`EMP__H3`),
against a 512 ceiling; with 4.5× headroom, nothing is truncated.

## Protocol

- Embeddings: `output_all_encoded_layers=True`, mask scatter-back to padded
  `(B, T, 768)`, mask-aware mean pooling and `h[:, 0]` for `[CLS]`. fp32.
- `StandardScaler` fit on train only, inside an sklearn `Pipeline`.
- Dev selects; **test is scored exactly once** per selected configuration.
- `train_model` restores the best-dev checkpoint, not the last epoch.
- Layer ranking averages over **both** poolings, so the mean-vs-`[CLS]`
  comparison is not confounded by depth.
- LR grid: `C ∈ logspace(-5, 3, 9)` × `class_weight ∈ {none, balanced}`.
- MLP: `TaperedMLP(768 → 256 → 2)`, dropout 0.1, AdamW lr 1e-3, weight decay
  0.01, batch 32, max 200 epochs, patience 10, 3 seeds (42/43/44).

### Caveats

- **The dev column is a selection statistic, not a performance estimate.** Each
  dataset receives 9 C × 2 class_weight × 12 layers × 2 poolings = 432 dev
  evaluations, and taking the max over those is upward-biased. Test is clean.
- The MLP uses an unweighted `CrossEntropyLoss` (the notebook's definition)
  while LR searches `class_weight`. At 0.44–0.59 positive fraction this
  asymmetry is cosmetic, but the two arms are not matched on it.
- The paper fine-tunes end-to-end and averages 3 seeds; **Table 6** is the
  reference used here. Table 9 disagrees with it on TF-H and TF-M.
- Assumption diagnostics (calibration, residual structure) were not assessed.

## Known issues from the original notebook

Applied:
- `output_all_encoded_layers=True` — HF's `output_hidden_states` lands in
  `**kwargs` on MosaicBERT and is silently ignored. Requires `model.pooler = None`
  and a scatter-back from the unpadded tensors. **Verified bit-identical** at
  layer 12 to the model's own default padded forward pass, for both poolings
  (`max_abs_diff = 0.0`; `scripts/01_verify_layers.py`).
- `class_weight=None` persisted as the literal `"none"`, so `pd.read_csv` no
  longer reads it as NaN.
- C grid widened from `logspace(-4, 2, 7)`; grid-edge selections fell from 14 %
  to 3 % (19 of 624). `in_notebook_grid` flags the original 7 values.

Not applicable to this selection, checked and recorded:
- **The suspected sequence-pair bug does not exist.** `tok([[a, b]])` and the
  documented `tok([a], [b])` return identical encodings, with `[SEP]` at the pair
  boundary *and* at the end — the fast tokenizer reads a list of 2-element lists
  as `batch_text_or_text_pairs`. See `results/epi_sep_check.txt`. Separately, no
  in-scope dataset is a pair task.
- GUE/GUE+ mixing — resolved by the dataset selection.
- 512-token truncation — the longest sampled sequence is 115 tokens (see Scope).
- Covid F1 — covid excluded; Table 12 confirms MCC for all five in-scope tasks.

## Outputs

**Input** files are primary evidence from the sweeps. **Derived** files are
rebuilt from the inputs by `06_build_tables.py` in under a minute.

| File | Kind | Contents |
|---|---|---|
| `results/lr_layer_sweep.csv` | input | 11,232 rows — the complete 12-layer LR grid |
| `results/mlp_results.csv` | input | 468 rows — per-seed MLP runs |
| `results/selected_layers.csv` | input | the three layers per task fed to the MLP arm |
| `results/paper_table6_dnabert2.csv` | input | the paper's fine-tuned DNABERT-2 MCC (Table 6) |
| `results/history_table.csv` | derived | 3,276 rows — the full history table (2,808 LR + 468 MLP) |
| `results/history_table_aggregated.csv` | derived | 312 rows — seed-aggregated / cell-best companion |
| `results/best_frozen_vs_paper.csv` | derived | 26 rows — best dev-selected frozen config per dataset vs. Table 6 |
| `results/best_vs_paper_with_finetune.csv` | derived | written only once `lora_results.csv` exists — frozen best and LoRA side by side vs. Table 6 |
| `results/mlp_batch_benchmark.csv` | record | batch 32 vs 256, 3 seeds |
| `results/verify_layers.json` | record | layer-12 equivalence check and sampled token lengths |
| `results/epi_sep_check.txt` | record | `[SEP]` placement for a sequence pair |
| `layer_embed/` | local | 1,950 `.npy` files, 53 GB — regenerable in ~10 min; not committed |

The sweep scripts also write per-epoch training curves
(`results/*_histories.csv.gz`). They are local diagnostics and are not committed.

### History table columns

`model_type`, `task`, `dataset`, `pool`, `layer`, `layer_role`, `seed`, `C`,
`class_weight`, `dev_mcc`, `test_mcc`, `dev_selected`, `best_epoch`,
`epochs_run`, `early_stopped`, `batch_size`, `hidden_dims`, `dropout`, `lr`,
`weight_decay`, `optimizer`, `max_epochs`, `patience`, `n_iter`,
`at_grid_edge`, `in_notebook_grid`, `hyperparams`.

LR rows span the full 18-point grid at each selected layer with `test_mcc`
populated only on the dev-selected row; MLP rows carry one row per seed.

LoRA rows (once `results/lora_results.csv` exists) use sentinels, because
fine-tuning has no readout layer and no free pooling choice: `layer = 0` (real
encoder layers are 1–12), `layer_role = "finetuned"` and `pool = "cls-trained"`
(the stock classification head reads position 0 through a pooler that LoRA
trains). LoRA settings (`r`, `alpha`, `lora_dropout`, `target_modules`) go in
`hyperparams`.

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
python scripts/06_build_tables.py      # <1 min, rebuilds the derived tables
```

`requirements.lock.txt` pins the full dependency closure at the versions that
produced `results/` (`pip install -r requirements.lock.txt` before
`pip install -e .` for an exact environment). Install editable: the package
resolves its default paths from the repo root and refuses to run from a
non-editable install. After the first run, `HF_HUB_OFFLINE=1` skips the
HuggingFace check.

`02_embed_all.py`, `03_lr_sweep.py` and `04_mlp_sweep.py` all resume from
partial state. `03_lr_sweep.py` runs `cpu_count // 3` parallel jobs by default
(6 jobs × `OMP_NUM_THREADS=3` was measured best on 20 cores); set
`SWEEP_N_JOBS` to override. Timings were measured on an RTX 4080 / 20 CPU cores.

### Paths

With the default layout nothing needs to be set: GUE goes in `data/GUE`, and
embeddings in `layer_embed/`, both gitignored. The variables are an escape
hatch for data that must live elsewhere (cluster scratch, another volume).

| Variable | Default | Meaning |
|---|---|---|
| `DNABERT2_GUE_ROOT` | `data/GUE` | GUE benchmark root (contains `EMP/`, `mouse/`, `prom/`, `tf/`) |
| `DNABERT2_EMBED_DIR` | `layer_embed` | cached per-layer embeddings (53 GB) |
| `DNABERT2_RESULTS_DIR` | `results` | sweep outputs and derived tables |
| `DNABERT2_FIGURES_DIR` | `figures` | figure output |
