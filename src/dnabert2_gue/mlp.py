"""MLP probe over the cached embeddings (verbatim from the original notebook).

Every torch import is inside a function body so that importing the package
does not require torch.
"""

from .config import DATASET_TASK, EMBED_DIR, SPLITS
from .embed import load_split

# --------------------------------------------------------------------------
# MLP (verbatim from the notebook)
# --------------------------------------------------------------------------

def _torch_bits():
    import torch
    import torch.nn as nn
    return torch, nn


class TaperedMLP:
    """Placeholder replaced at import time by the real nn.Module subclass."""


def _build_mlp_class():
    torch, nn = _torch_bits()

    class _TaperedMLP(nn.Module):
        """MLP classifier with a configurable number of hidden layers and dropout.

        Each hidden layer is Linear -> ReLU -> Dropout. The output layer is a
        plain Linear layer, so the model returns raw logits (no softmax);
        use with nn.CrossEntropyLoss, and argmax over logits for predictions.
        """
        def __init__(self, in_features, num_classes, dims=(256, 64), dropout=0.1):
            super().__init__()
            layers, prev = [], in_features
            for d in dims:
                layers += [nn.Linear(prev, d), nn.ReLU(), nn.Dropout(dropout)]
                prev = d
            layers.append(nn.Linear(prev, num_classes))    # raw logits out
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return _TaperedMLP


def prep_data(dataset, pool, layer, d=EMBED_DIR):
    """Load splits, standardize X with train stats, return float/long tensors."""
    import torch
    from sklearn.preprocessing import StandardScaler

    splits = {s: load_split(dataset, s, layer, pool, d) for s in SPLITS}
    scaler = StandardScaler().fit(splits["train"][0])
    out = {}
    for s, (X, y) in splits.items():
        out[s] = (torch.from_numpy(scaler.transform(X)).float(),
                  torch.from_numpy(y).long())
    return out


def evaluate(model, X, y, criterion, device):
    """Return (loss, mcc) on a full split; no gradients, dropout off."""
    import torch
    from sklearn.metrics import matthews_corrcoef

    model.eval()
    with torch.no_grad():
        logits = model(X.to(device))
        loss = criterion(logits, y.to(device)).item()
        preds = logits.argmax(dim=1).cpu().numpy()
    return loss, matthews_corrcoef(y.numpy(), preds)


def train_model(model, train_loader, X_dv, y_dv, criterion, optimizer, device,
                max_epochs=200, patience=10, debug=False):
    """Train with early stopping on dev MCC; restore the best state, not the last."""
    import copy

    best_mcc, best_state, best_epoch, bad_epochs = -float("inf"), None, 0, 0
    history = []

    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_X.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        dev_loss, dev_mcc = evaluate(model, X_dv, y_dv, criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "dev_loss": dev_loss, "dev_mcc": dev_mcc})
        if debug:
            print(epoch, train_loss, dev_loss, dev_mcc, flush=True)

        if dev_mcc > best_mcc:
            best_mcc, best_epoch, bad_epochs = dev_mcc, epoch + 1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, history


# Hyperparameters held fixed across the MLP sweep (notebook values).
MLP_DIMS = (256,)
MLP_DROPOUT = 0.1
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 0.01
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 10


def run_mlp(dataset, pool, layer, seed, batch_size=32, d=EMBED_DIR,
            device=None, debug=False):
    """One MLP run; returns a result dict plus the per-epoch history."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    data = prep_data(dataset, pool, layer, d)
    X_tr, y_tr = data["train"]
    X_dv, y_dv = data["dev"]
    X_te, y_te = data["test"]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr),
                              batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    cls = _build_mlp_class()
    model = cls(X_tr.shape[1], int(len(torch.unique(y_tr))),
                dims=MLP_DIMS, dropout=MLP_DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=MLP_LR,
                            weight_decay=MLP_WEIGHT_DECAY)

    model, best_epoch, history = train_model(
        model, train_loader, X_dv, y_dv, criterion, optimizer, device,
        max_epochs=MLP_MAX_EPOCHS, patience=MLP_PATIENCE, debug=debug)

    _, dev_mcc = evaluate(model, X_dv, y_dv, criterion, device)
    _, test_mcc = evaluate(model, X_te, y_te, criterion, device)
    return {
        "dataset": dataset, "task": DATASET_TASK[dataset], "pool": pool,
        "layer": layer, "seed": seed, "batch_size": batch_size,
        "best_epoch": best_epoch, "epochs_run": len(history),
        "early_stopped": len(history) < MLP_MAX_EPOCHS,
        "dev_mcc": dev_mcc, "test_mcc": test_mcc,
        "hidden_dims": "x".join(str(x) for x in MLP_DIMS),
        "dropout": MLP_DROPOUT, "lr": MLP_LR,
        "weight_decay": MLP_WEIGHT_DECAY, "optimizer": "AdamW",
        "max_epochs": MLP_MAX_EPOCHS, "patience": MLP_PATIENCE,
    }, history
