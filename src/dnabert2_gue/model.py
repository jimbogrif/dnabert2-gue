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
    import torch
    from transformers import AutoTokenizer, AutoModel

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
    import torch

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
