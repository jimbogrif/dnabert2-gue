"""LoRA fine-tuning arm -- STUB, not implemented.

The third arm of the comparison: instead of probing frozen embeddings, fine-tune
DNABERT-2 with LoRA adapters and a trained classification head. To be
hand-written; this file fixes the interface and records what is already known.

Implementation notes (learned the hard way, do not rediscover):

- ``target_modules=["Wqkv"]`` is correct. MosaicBERT's attention is a *fused*
  ``attention.self.Wqkv``; there are no ``query``/``key``/``value`` modules, so a
  peft config copied from a BERT example matches nothing and silently trains
  only the head. Other available targets: ``attention.output.dense``,
  ``mlp.gated_layers``, ``mlp.wo``.
- If extending ``target_modules``, use the explicit ``attention.output.dense``,
  not bare ``"dense"`` -- peft matches on suffix and ``"dense"`` would also hit
  ``pooler.dense``, which belongs in ``modules_to_save``.
- ``modules_to_save=["classifier", "pooler"]`` is required. The checkpoint
  ships no trained pooler, so leaving it frozen at random init would cripple
  the arm.
- Apply the Triton fix to the ``AutoModelForSequenceClassification`` module
  before the first forward pass (same one-liner as ``model.load_model``:
  ``sys.modules[model.__class__.__module__].flash_attn_qkvpacked_func = None``).
- Operating point on an RTX 4080: batch 128 with bf16 autocast, ~817 seq/s
  training, 6.4 GiB peak. Throughput does not improve past 128 while memory
  grows 1.7x. Do not copy the MLP arm's batch 32 -- the batch benchmark found
  bs32 and bs256 indistinguishable (0.2911 +/- 0.0043 vs 0.2909 +/- 0.0148),
  so nothing rides on matching it and 128 is ~2x faster.
- No EPI/pair handling is needed. All 26 in-scope datasets are 2-column
  (``sequence``, ``label``), and ``results/epi_sep_check.txt`` established the
  suspected pair-tokenization bug does not exist.
- Pooling is not a free choice once you fine-tune: the stock SEQ_CLS head reads
  position 0 through the (trained) pooler. Rows are labelled
  ``pool="cls-trained"`` -- a direct test of the README's explanation for why
  mean beat [CLS] in the frozen arm (the checkpoint has no trained pooler).

Schema contract -- one row per run in ``results/lora_results.csv``, mirroring
``mlp_results.csv``::

    dataset, task, pool, layer, seed, batch_size, best_epoch, epochs_run,
    early_stopped, dev_mcc, test_mcc, hidden_dims, dropout, lr, weight_decay,
    optimizer, max_epochs, patience, wall_s,
    trainable_params, total_params, pct_trainable,
    lora_r, lora_alpha, lora_dropout, target_modules

with ``pool="cls-trained"``, ``layer=0`` (sentinel: fine-tuning has no readout
layer; real layers are 1-12, and NaN would be silently dropped by groupby) and
``hidden_dims`` blank. ``wall_s`` and the run keys are attached by
``scripts/05_lora_sweep.py``, as ``04_mlp_sweep.py`` does for ``run_mlp``.
``scripts/06_build_tables.py`` reads this file if it exists.

TODO when this lands: add ``trainable_params`` and ``pct_trainable`` to
``COLS`` in ``scripts/06_build_tables.py`` (deferred so the frozen-only tables
keep their regression checksums until then).
"""


def run_lora(dataset, seed, lr, r=8, alpha=16, lora_dropout=0.05,
             batch_size=128, max_epochs=10, patience=3,
             device=None, debug=False):
    """One LoRA fine-tuning run; returns (result_dict, history).

    Mirrors ``mlp.run_mlp``: ``result_dict`` holds the columns of the schema
    contract above (minus ``wall_s``), and ``history`` is a list of per-epoch
    dicts (``epoch``, ``train_loss``, ``dev_loss``, ``dev_mcc``). There is no
    ``pool`` or ``layer`` argument -- fine-tuning has neither. Early stopping on
    dev MCC with the best state restored, as in ``mlp.train_model``.

    Raises:
        NotImplementedError: always. The LoRA arm is not implemented yet.
    """
    raise NotImplementedError(
        "run_lora is a stub: the LoRA arm is not implemented yet "
        "(see the dnabert2_gue.lora module docstring for the contract)")
