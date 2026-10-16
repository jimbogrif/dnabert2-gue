"""LoRA fine-tuning arm (ported from Jim's hand-written notebook).

The PEFT configuration, training loop, best-state checkpointing and evaluation
protocol are the notebook's. Deliberate changes, each so the arm joins the
frozen-probe tables on equal terms:

  - dataset identity from ``manifest()`` (so ``prom`` splits into PD / CPD);
  - ``lr`` is an explicit argument and a recorded column;
  - dev is evaluated once per epoch, not every 200 steps, so every run gets the
    same number of selection draws regardless of dataset size;
  - training stops on dev patience rather than at a fixed epoch count, matching
    the MLP arm's ``train_model``. A 5-epoch cap was tried first and bound on
    every calibration dataset, which is what a budget rather than a backstop
    looks like; ``LORA_MAX_EPOCHS`` is now the backstop and patience decides;
  - warmup is a ratio of total steps, not a fixed 50 steps;
  - bf16 autocast on CUDA at batch 128 (no GradScaler: bf16 has fp32's range);
  - parameter counts and matched modules are returned, not printed;
  - the EPI sequence-pair branch is dropped (no in-scope dataset is a pair).

Implementation notes (learned the hard way, do not rediscover):

- ``target_modules=["Wqkv"]`` is correct. MosaicBERT's attention is a *fused*
  ``attention.self.Wqkv``; there are no ``query``/``key``/``value`` modules, so a
  peft config copied from a BERT example matches nothing and silently trains
  only the head. ``build_lora_model`` asserts the match is non-empty.
- Never target bare ``"dense"`` -- peft matches on suffix and it would also hit
  ``pooler.dense``, which belongs in ``modules_to_save``. Use the explicit
  ``attention.output.dense`` if that target is ever wanted.
- ``modules_to_save=["classifier", "pooler"]`` is required. The checkpoint
  ships no trained pooler, so leaving it frozen at random init would cripple
  the arm. The load-time warning that they are newly initialized is expected.
- Pooling is not a free choice once you fine-tune: the SEQ_CLS head reads
  position 0 through the (trained) pooler, so rows carry
  ``pool="cls-trained"`` and the sentinel ``layer=0``.
"""

import os
import sys

import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from .config import (DATASET_TASK, LORA_ALPHA, LORA_BATCH, LORA_DROPOUT,
                     LORA_MAX_EPOCHS, LORA_MAX_LENGTH, LORA_PATIENCE, LORA_R,
                     LORA_VARIANTS, LORA_WARMUP_RATIO, LORA_WEIGHT_DECAY,
                     MODEL_NAME, SPLITS, manifest)

MODULES_TO_SAVE = ["classifier", "pooler"]

# Measured wiring for the attention-only variant at r=8, binary head:
# 294,912 adapter + 590,592 pooler + 1,538 classifier.
ATTN_TRAINABLE = 887_042
ATTN_TOTAL = 117_957_124

# Columns of results/lora_results.csv, in order (wall_s attached by the sweep).
RESULT_COLS = [
    "dataset", "task", "variant", "pool", "layer", "seed", "lr", "batch_size",
    "max_epochs", "patience", "epochs_run", "best_epoch", "early_stopped", "dev_mcc",
    "test_mcc", "lora_r", "lora_alpha", "lora_dropout", "target_modules",
    "modules_to_save", "trainable_params", "total_params", "pct_trainable",
    "warmup_steps", "warmup_ratio", "weight_decay", "optimizer", "precision",
    "max_length", "eval_cadence", "wall_s",
]


def _autocast(device):
    return torch.autocast(device.type, dtype=torch.bfloat16,
                          enabled=device.type == "cuda")


def warmup_steps(warmup_ratio, total_steps):
    return max(1, int(warmup_ratio * total_steps))


# Store GUE Sequence + Label CSVs as a map-style dataset (DataLoader needs only
# __len__/__getitem__, so no torch base class).
class SeqDataset:
    """Raw sequences + labels from a 2-column (sequence, label) split."""
    def __init__(self, df):
        self.a = df["sequence"].tolist()
        self.y = df["label"].tolist()
    def __len__(self):
        return len(self.y)
    def __getitem__(self, i):
        return self.a[i], self.y[i]


def make_collate(tok, max_length=LORA_MAX_LENGTH):
    """Tokenize a batch of (sequence, label) pairs for DataLoader."""
    def collate(batch):
        a, y = zip(*batch)
        enc = tok(list(a), return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length)
        return enc["input_ids"], enc["attention_mask"], torch.tensor(y, dtype=torch.long)
    return collate


# Create lora model:
def build_lora_model(n_classes, target_modules, r=LORA_R, alpha=LORA_ALPHA,
                     lora_dropout=LORA_DROPOUT, model_name=None):
    """Build a LoRA model on DNABERT-2 for sequence classification.

    LoRA-trains ``target_modules`` and fully trains the classifier head and the
    pooler. Returns (PeftModelForSequenceClassification,
    {'trainable_params', 'total_params', 'pct_trainable', 'matched_modules',
     'breakdown'}).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name or MODEL_NAME,
        num_labels=n_classes,
        trust_remote_code=True)
    sys.modules[model.__class__.__module__].flash_attn_qkvpacked_func = None
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=lora_dropout,
                     target_modules=list(target_modules),
                     modules_to_save=list(MODULES_TO_SAVE),   # train the whole head
                     task_type="SEQ_CLS")
    model = get_peft_model(model, cfg)

    trainable, total = model.get_nb_trainable_parameters()
    matched = [n for n, m in model.named_modules() if isinstance(m, LoraLayer)]
    assert matched, f"target_modules {target_modules} matched no module"

    breakdown = {"lora": 0, "pooler": 0, "classifier": 0, "other": 0}
    for n, p in model.named_parameters():
        if p.requires_grad:
            k = ("lora" if "lora_" in n else "pooler" if "pooler" in n
                 else "classifier" if "classifier" in n else "other")
            breakdown[k] += p.numel()

    if r == 8 and n_classes == 2:
        if list(target_modules) == ["Wqkv"]:
            assert (trainable, total) == (ATTN_TRAINABLE, ATTN_TOTAL), \
                f"attn wiring changed: {trainable:,} / {total:,} ({breakdown})"
        elif "Wqkv" in target_modules:
            assert trainable > ATTN_TRAINABLE, \
                f"{target_modules} trains no more than attn alone ({breakdown})"

    return model, {"trainable_params": int(trainable), "total_params": int(total),
                   "pct_trainable": 100 * trainable / total,
                   "matched_modules": matched, "breakdown": breakdown}


# Training loop:
def train_lora(model,
               train_loader,
               dev_loader,
               device,
               lr,
               epochs=LORA_MAX_EPOCHS,
               patience=LORA_PATIENCE,
               warmup_ratio=LORA_WARMUP_RATIO,
               weight_decay=LORA_WEIGHT_DECAY,
               debug=False):
    """ Train a PEFT model on a train split, evaluating dev once per epoch.

    Stops after ``patience`` consecutive epochs without a dev improvement, as
    the MLP arm's ``train_model`` does; ``epochs`` is a backstop, not a budget.
    The LR schedule is still built over ``len(train_loader) * epochs``, so an
    early-stopped run simply never reaches the tail of the decay — the same
    behaviour early stopping has in any warmup+decay setup.

    Return (model with best-dev state restored, best MCC, best epoch, history,
    early_stopped).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    sched = get_linear_schedule_with_warmup(
        opt, warmup_steps(warmup_ratio, total_steps), total_steps)
    crit = nn.CrossEntropyLoss()
    best_mcc, best_epoch, history = -float("inf"), 0, []
    stale, early_stopped = 0, False
    # Checkpoint before lora in case dataset is tiny
    best_state = {n: p.detach().cpu().clone()
                  for n, p in model.named_parameters() if p.requires_grad}

    for epoch in range(epochs):
        running_loss = torch.zeros((), device=device)
        for ids, mask, y in train_loader:
            model.train()
            with _autocast(device):
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device))[0]
                loss = crit(logits, y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            running_loss += loss.detach().float() * y.size(0)

        train_loss = running_loss.item() / len(train_loader.dataset)
        mcc = evaluate(model, dev_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "dev_mcc": mcc})
        if mcc > best_mcc:   # save only the trainable weights (small)
            best_mcc, best_epoch, stale = mcc, epoch + 1, 0
            best_state = {n: p.detach().cpu().clone()
                          for n, p in model.named_parameters() if p.requires_grad}
        else:
            stale += 1
        if debug:
            print(f"epoch {epoch + 1}  loss {train_loss:.4f}  dev MCC {mcc:.4f}"
                  f"{'' if stale == 0 else f'  (stale {stale}/{patience})'}", flush=True)
        if stale >= patience:
            early_stopped = True
            break

    model.load_state_dict(best_state, strict=False)   # restore best adapters + head
    return model, best_mcc, best_epoch, history, early_stopped


# Evaluation protocol
def evaluate(model, loader, device):
    model.eval()
    preds, ys = [], []
    with torch.no_grad(), _autocast(device):
        for ids, mask, y in loader:                       # batched: the full split won't fit
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device))[0]
            preds.append(logits.argmax(1).cpu())
            ys.append(y)
    return matthews_corrcoef(torch.cat(ys), torch.cat(preds))


def run_lora(dataset, seed, lr, variant="attn",
             r=LORA_R, alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
             batch_size=LORA_BATCH, max_epochs=LORA_MAX_EPOCHS,
             patience=LORA_PATIENCE,
             warmup_ratio=LORA_WARMUP_RATIO, weight_decay=LORA_WEIGHT_DECAY,
             max_length=LORA_MAX_LENGTH, device=None, debug=False):
    """One LoRA fine-tuning run; returns (result_dict, history).

    Mirrors ``mlp.run_mlp``: the result dict holds the ``RESULT_COLS`` columns
    except ``wall_s``, which the sweep attaches. No ``pool``/``layer`` argument
    -- fine-tuning has neither. ``device=None`` auto-detects; bf16 autocast is
    used only on CUDA, and ``precision`` records what actually ran.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    target_modules = LORA_VARIANTS[variant]

    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)

    path = {key: p for _, key, p in manifest()}[dataset]
    df_tr, df_dev, df_test = (pd.read_csv(os.path.join(path, f"{s}.csv")) for s in SPLITS)
    n_classes = int(df_tr["label"].nunique())
    assert n_classes == 2, f"{dataset}: expected a binary task, got {n_classes} classes"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    collate = make_collate(tok, max_length)
    train_loader = DataLoader(SeqDataset(df_tr), batch_size=batch_size, shuffle=True,
                              collate_fn=collate, generator=g, num_workers=0)
    dev_loader = DataLoader(SeqDataset(df_dev), batch_size=batch_size * 2,
                            collate_fn=collate, num_workers=0)
    test_loader = DataLoader(SeqDataset(df_test), batch_size=batch_size * 2,
                             collate_fn=collate, num_workers=0)

    model, info = build_lora_model(n_classes, target_modules, r, alpha, lora_dropout)
    model = model.to(device)
    try:
        model, dev_mcc, best_epoch, history, early_stopped = train_lora(
            model, train_loader, dev_loader, device, lr, epochs=max_epochs,
            patience=patience, warmup_ratio=warmup_ratio,
            weight_decay=weight_decay, debug=debug)
        test_mcc = evaluate(model, test_loader, device)   # scored once, best-dev state
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "dataset": dataset, "task": DATASET_TASK[dataset], "variant": variant,
        "pool": "cls-trained", "layer": 0, "seed": seed, "lr": lr,
        "batch_size": batch_size, "max_epochs": max_epochs,
        "patience": patience,
        "epochs_run": len(history), "best_epoch": best_epoch,
        "early_stopped": early_stopped,   # False here means the cap bound
        "dev_mcc": float(dev_mcc), "test_mcc": float(test_mcc),
        "lora_r": r, "lora_alpha": alpha, "lora_dropout": lora_dropout,
        "target_modules": "|".join(target_modules),
        "modules_to_save": "|".join(MODULES_TO_SAVE),
        "trainable_params": info["trainable_params"],
        "total_params": info["total_params"],
        "pct_trainable": info["pct_trainable"],
        "warmup_steps": warmup_steps(warmup_ratio, len(train_loader) * max_epochs),
        "warmup_ratio": warmup_ratio, "weight_decay": weight_decay,
        "optimizer": "AdamW",
        "precision": "bf16" if device.type == "cuda" else "fp32",
        "max_length": max_length, "eval_cadence": "per_epoch",
    }, history
