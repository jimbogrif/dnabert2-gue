"""LoRA arm: fine-tune DNABERT-2 on each in-scope dataset -- STUB, not implemented.

Planned shape, same as 04_mlp_sweep.py:

- plan = 26 datasets x 3 seeds (42, 43, 44) = 78 runs of
  ``dnabert2_gue.lora.run_lora(dataset, seed, lr)``;
- per-run rows to results/lora_results.csv (schema in the dnabert2_gue.lora
  docstring), resuming by reading it and building
  ``done = set(zip(dataset, seed))``; flush every 20 runs;
- per-epoch curves to results/lora_histories.csv.gz (local diagnostics --
  the evidence for whether the epoch budget sufficed; gitignored);
- attach ``wall_s`` and the run keys to each result here, not in run_lora.

Sizing: 627,671 train rows across the 26 datasets; at ~817 seq/s one seed over
all 26 is ~40 min at 3 epochs, so 78 runs is 2.0-3.3 h. Learning rate is not
tuned per dataset (a 3-point grid would be 11.5 h): calibrate once, ~30 min,
on mouse__3, prom__prom_core_tata and EMP__H3K4me3 over lr in
{1e-3, 3e-4, 1e-4}, then run the frozen protocol for all 26. That is insurance
against a bad-lr false negative and keeps the tuning budget comparable to the
MLP arm's fixed protocol.

Then rerun 06_build_tables.py, which picks up lora_results.csv and writes
results/best_vs_paper_with_finetune.csv.

Usage: 05_lora_sweep.py lr
"""

import sys

import dnabert2_gue as SL  # noqa: F401

SEEDS = (42, 43, 44)

if __name__ == "__main__":
    raise NotImplementedError(
        "05_lora_sweep.py is a stub: the LoRA arm is not implemented yet "
        "(see the module docstring and dnabert2_gue.lora)")
