"""DNABERT-2 loading and all-layer embedding.

Two of the three documented fixes relative to the original notebook live here:

  1. ``model.pooler = None`` + ``output_all_encoded_layers=True`` so every
     encoder layer is available (HF's ``output_hidden_states`` lands in
     ``**kwargs`` on MosaicBERT and is silently ignored).
  2. The returned per-layer tensors are *unpadded*; they are scattered back
     to (B, T, 768) with the attention mask before pooling, which reproduces
     the notebook's padded output exactly.
"""

import sys

import torch
from transformers import AutoModel, AutoTokenizer

from .config import MODEL_NAME

# --------------------------------------------------------------------------
# Model loading and all-layer embedding
# --------------------------------------------------------------------------

def load_model(device=None):
    """DNABERT-2 with the Triton fix and the pooler removed.

    The pooler must go: with output_all_encoded_layers=True it receives the
    unpadded tensor and raises a shape error. It is untrained anyway -- the
    checkpoint has no pooler weights.

    ``device=None`` auto-detects (cuda if available, else cpu).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # Resolved from module globals at call time, so this selects the pytorch
    # attention branch without disabling Triton process-wide.
    sys.modules[model.__class__.__module__].flash_attn_qkvpacked_func = None
    model.pooler = None
    return tok, model.to(device).eval()


def embed_batch_layers(seqs, tok, model, max_length=512, device=None):
    """All 12 layers of one batch, pooled.

    Returns (mean, cls) each shaped (n_layers, B, 768) as float32 cpu tensors,
    plus the per-sequence token count. ``device=None`` auto-detects.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = tok(seqs, return_tensors="pt", padding=True,
              truncation=True, max_length=max_length).to(device)
    attn = enc["attention_mask"]
    mask = attn.bool()
    layers = model(enc["input_ids"], attention_mask=attn,
                   output_all_encoded_layers=True)[0]

    m = attn.unsqueeze(-1).float()
    denom = m.sum(dim=1)
    B, T = mask.shape
    means, clss = [], []
    for flat in layers:
        h = torch.zeros(B, T, flat.shape[-1], device=flat.device, dtype=flat.dtype)
        h[mask] = flat
        means.append(((h.float() * m).sum(dim=1) / denom).cpu())
        clss.append(h[:, 0].float().cpu())
    return torch.stack(means), torch.stack(clss), attn.sum(dim=1).cpu().numpy()


# --------------------------------------------------------------------------
# Positional readout: overlap-weighted pooling into fixed base-pair windows
# --------------------------------------------------------------------------

def window_weights(offsets, lengths, n_windows):
    """(B, W, T) base-pair overlap weights, normalized within each window.

    Each sequence is tiled into ``n_windows`` equal-width bp windows and every
    token contributes to a window in proportion to how many of its bases fall
    inside it. The weighting is what makes this well defined for BPE: tokens
    average 4.76 bp on GUE splice and do not align to window boundaries, so a
    token straddling one belongs partly to both.

    ``offsets`` is the fast tokenizer's (B, T, 2) offset_mapping. Special and
    padding tokens carry (0, 0) and so have zero width, which drops them with
    no special-casing -- asserted below rather than assumed.

    Windows tile [0, L) and tokens tile [0, L), so summing a token's overlaps
    across windows returns its width. That makes the bp-weighted average of
    the W window vectors exactly the bp-weighted mean over tokens, i.e. this
    readout strictly contains whole-sequence mean pooling.
    """
    off = offsets.float()
    s, e = off[..., 0], off[..., 1]                              # (B, T)
    L = lengths.float().unsqueeze(1)                             # (B, 1)
    frac = torch.linspace(0, 1, n_windows + 1, device=off.device)
    edges = L * frac.unsqueeze(0)                                # (B, W+1)
    lo, hi = edges[:, :-1].unsqueeze(2), edges[:, 1:].unsqueeze(2)   # (B, W, 1)
    overlap = (torch.minimum(hi, e.unsqueeze(1))
               - torch.maximum(lo, s.unsqueeze(1))).clamp(min=0)     # (B, W, T)
    total = overlap.sum(dim=2, keepdim=True)
    return overlap / total.clamp(min=1e-9), total.squeeze(2)


def embed_batch_positional(seqs, tok, model, n_windows, max_length=512,
                           device=None):
    """One forward pass -> mean, cls and window-pooled readouts.

    Returns (mean, cls, win, n_tokens, empty_windows) with ``mean``/``cls``
    shaped (n_layers, B, 768) and ``win`` (n_layers, B, W, 768), all float32
    on cpu. ``empty_windows`` counts (sequence, window) pairs that received no
    token mass, which can only happen if a sequence is truncated.

    ``mean`` and ``cls`` are computed by the same expressions as
    ``embed_batch_layers`` so the two paths can be asserted equal.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not getattr(tok, "is_fast", False):
        raise RuntimeError(
            "window pooling needs offset mappings, which require a fast "
            "tokenizer; AutoTokenizer returned a slow one for " + MODEL_NAME)
    enc = tok(seqs, return_tensors="pt", padding=True, truncation=True,
              max_length=max_length, return_offsets_mapping=True)
    off = enc.pop("offset_mapping").to(device)
    enc = enc.to(device)
    attn = enc["attention_mask"]
    mask = attn.bool()

    widths = (off[..., 1] - off[..., 0])
    assert int((widths < 0).sum()) == 0, "negative token width in offset_mapping"
    lengths = torch.tensor([len(s) for s in seqs], device=device)
    assert int((off[..., 1].max(dim=1).values > lengths).sum()) == 0, \
        "token offset runs past the end of its sequence"

    w, total = window_weights(off, lengths, n_windows)            # (B, W, T)
    layers = model(enc["input_ids"], attention_mask=attn,
                   output_all_encoded_layers=True)[0]

    m = attn.unsqueeze(-1).float()
    denom = m.sum(dim=1)
    B, T = mask.shape
    means, clss, wins = [], [], []
    for flat in layers:
        h = torch.zeros(B, T, flat.shape[-1], device=flat.device, dtype=flat.dtype)
        h[mask] = flat
        hf = h.float()
        means.append(((hf * m).sum(dim=1) / denom).cpu())
        clss.append(h[:, 0].float().cpu())
        wins.append(torch.bmm(w, hf).cpu())                       # (B, W, 768)
    return (torch.stack(means), torch.stack(clss), torch.stack(wins),
            attn.sum(dim=1).cpu().numpy(), int((total <= 0).sum()))


def shuffle_windows(X, n_windows, seed):
    """Permute the W window blocks of a flattened (n, W*768) matrix per row.

    The dimension control for the positional arm: identical content and
    identical feature count, position destroyed. Derived from the stored win
    cache at probe time rather than cached separately.
    """
    import numpy as np
    n, d = X.shape
    assert d % n_windows == 0, (d, n_windows)
    blocks = X.reshape(n, n_windows, d // n_windows)
    rng = np.random.default_rng(seed)
    perm = np.argsort(rng.random((n, n_windows)), axis=1)
    out = np.take_along_axis(blocks, perm[:, :, None], axis=1)
    return out.reshape(n, d)
