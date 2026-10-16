"""Correctness gate for the all-layer embedding path.

1. Embed a small split with output_all_encoded_layers=True + scatter-back and
   check that layer 12 is bit-identical to the model's own default padded
   forward pass (an independently loaded reference model, pooler kept, pooled
   the same way). Needs no cached arrays.
2. Token-count headroom: tokenize a 200-sequence sample from each of the 26
   training splits and record the longest, against the 512 ceiling.
3. Record where [SEP] actually lands for a sequence pair under the notebook's
   list-of-lists call vs. the documented tok(list_a, list_b) call. No pair
   dataset is in scope for this sweep; this just writes the answer down. EPI
   is GUE+, so this check is skipped (and recorded as skipped) if absent.
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel

import dnabert2_gue as SL

OUT = os.path.join(SL.RESULTS_DIR, "verify_layers.json")
SEP_OUT = os.path.join(SL.RESULTS_DIR, "epi_sep_check.txt")
EPI_DEV = os.path.join(SL.GUE_ROOT, "EPI", "GM12878", "dev.csv")

if not torch.cuda.is_available():
    sys.exit("01_verify_layers.py needs a CUDA GPU (>=6 GB VRAM). The committed "
             "results/verify_layers.json records the outcome on the reference "
             "machine.")

tok, model = SL.load_model()
report = {"torch": torch.__version__, "cuda": torch.cuda.is_available()}

# Reference: the stock model in default mode, loaded independently of
# load_model() because that one drops the pooler (which raises on unpadded
# tensors). Only the Triton fix is applied.
ref_model = AutoModel.from_pretrained(SL.MODEL_NAME, trust_remote_code=True)
sys.modules[ref_model.__class__.__module__].flash_attn_qkvpacked_func = None
ref_model = ref_model.to("cuda").eval()


def reference_pooled(seqs):
    """Last hidden state from the default padded forward, mean- and CLS-pooled."""
    enc = tok(seqs, return_tensors="pt", padding=True,
              truncation=True, max_length=512).to("cuda")
    attn = enc["attention_mask"]
    h = ref_model(enc["input_ids"], attention_mask=attn)[0]        # (B, T, 768)
    m = attn.unsqueeze(-1).float()
    mean = ((h.float() * m).sum(dim=1) / m.sum(dim=1)).cpu()
    return mean, h[:, 0].float().cpu()


# --- 1. layer-12 equivalence -------------------------------------------------
checks = []
for dataset, split in [("mouse__3", "dev"), ("prom__prom_core_tata", "dev")]:
    cat, name = dataset.split("__", 1)
    df = pd.read_csv(os.path.join(SL.GUE_ROOT, cat, name, f"{split}.csv"))
    seqs = df["sequence"].tolist()

    means, clss, ref_means, ref_clss = [], [], [], []
    with torch.inference_mode():
        for i in range(0, len(seqs), 64):
            m, c, _ = SL.embed_batch_layers(seqs[i:i + 64], tok, model)
            means.append(m)
            clss.append(c)
            rm, rc = reference_pooled(seqs[i:i + 64])
            ref_means.append(rm)
            ref_clss.append(rc)
    mean12 = torch.cat(means, dim=1)[SL.FINAL_LAYER - 1].numpy()
    cls12 = torch.cat(clss, dim=1)[SL.FINAL_LAYER - 1].numpy()
    ref_mean = torch.cat(ref_means).numpy()
    ref_cls = torch.cat(ref_clss).numpy()

    for pool, got, ref in [("mean", mean12, ref_mean), ("cls", cls12, ref_cls)]:
        d = np.abs(got - ref)
        checks.append({
            "dataset": dataset, "split": split, "pool": pool,
            "shape_ref": list(ref.shape), "shape_got": list(got.shape),
            "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
            "ref_abs_mean": float(np.abs(ref).mean()),
            "exact": bool(np.array_equal(got, ref)),
            "allclose_1e4": bool(np.allclose(got, ref, atol=1e-4, rtol=0)),
        })
        print(checks[-1], flush=True)
report["layer12_vs_default_forward"] = checks
del ref_model

# --- 2. token-count headroom -------------------------------------------------
lens = {}
for task, dataset, path in SL.manifest():
    df = pd.read_csv(os.path.join(path, "train.csv"))
    s = df["sequence"].sample(min(200, len(df)), random_state=0).tolist()
    n = [len(tok(x)["input_ids"]) for x in s]
    lens[dataset] = {"task": task, "n_tok_min": int(np.min(n)),
                     "n_tok_med": float(np.median(n)), "n_tok_max": int(np.max(n)),
                     "bp_max": int(df["sequence"].str.len().max())}
report["token_lengths"] = lens
report["max_tokens_any_dataset"] = int(max(v["n_tok_max"] for v in lens.values()))

# --- 3. EPI [SEP] placement --------------------------------------------------
if os.path.exists(EPI_DEV):
    epi = pd.read_csv(EPI_DEV)
    a, b = str(epi.iloc[0, 0]), str(epi.iloc[0, 1])
    lines = [f"EPI/GM12878 dev row 0: len(a)={len(a)} len(b)={len(b)}",
             f"columns: {list(epi.columns)}", ""]
    for label, call in [
        ("notebook form  tok([[a, b]])", lambda: tok([[a, b]], padding=True,
                                                     truncation=True, max_length=512)),
        ("documented     tok([a], [b])", lambda: tok([a], [b], padding=True,
                                                     truncation=True, max_length=512)),
    ]:
        try:
            ids = call()["input_ids"][0]
            toks = tok.convert_ids_to_tokens(ids)
            sep = [i for i, t in enumerate(toks) if t == "[SEP]"]
            lines += [f"{label}: n_tokens={len(ids)}  [SEP] at {sep}  "
                      f"(last index {len(ids) - 1})",
                      f"    head: {toks[:4]}",
                      f"    tail: {toks[-4:]}", ""]
        except Exception as e:                                     # noqa: BLE001
            lines += [f"{label}: RAISED {type(e).__name__}: {e}", ""]
    sep_text = "\n".join(lines)
    print(sep_text, flush=True)
    report["epi_sep_check"] = {"skipped": False}
else:
    sep_text = None
    report["epi_sep_check"] = {"skipped": True,
                               "reason": "EPI (GUE+) not present under GUE_ROOT"}
    print("EPI not found under GUE_ROOT; [SEP] check skipped", flush=True)

os.makedirs(SL.RESULTS_DIR, exist_ok=True)
if sep_text is not None:
    with open(SEP_OUT, "w") as fh:
        fh.write(sep_text)
with open(OUT, "w") as fh:
    json.dump(report, fh, indent=2)
print("wrote", OUT, SEP_OUT if sep_text is not None else "")

ok = all(c["exact"] for c in checks)
print("layer-12 equivalence:", "PASS" if ok else "FAIL", flush=True)
sys.exit(0 if ok else 1)
