"""
train_slm_rul.py  (v7 - Definitive)
======================================
SLM-based RUL prediction on NASA battery HIS data.

DEFINITIVE DATA STRATEGY:
  The B00XX_synthetic+Real.csv contains:
    - 6000 synthetic rows (first N_SYNTH rows)
    - ~168 real rows (last rows, from original measurements)

  The REAL rows form the actual monotonically-degrading time series.
  The SYNTHETIC rows share the same feature distributions but are NOT
  a real temporal sequence (they are Gaussian samples with cycle numbers
  that do NOT represent a true degradation trajectory).

  Strategy:
    1. Extract real rows (~168) sorted by cycle -> true time series
    2. Compute RUL on full real sequence (1.0 -> 0.0)
    3. 80/20 split on real cycles (train: first 80%, test: last 20%)
    4. For TRAINING only: generate augmented windows by jittering
       the real training features with small Gaussian noise
       (repeats each window N_AUG times with slight variations)
    5. Evaluate on REAL test cycles only (no augmentation)

This gives:
  - ~134 real train cycles -> ~124 base windows
  - x N_AUG augmentations = thousands of training samples
  - 34 real test cycles -> 25 test windows for honest evaluation
"""

import os, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

from slm_rul_model import SLMTransformerRUL, RULDataset

# ── Config ────────────────────────────────────────────────────────────────────
HIS_DIR      = r"d:\research paper - Copy\code+data\code+data\dataset\HIS"
BATTERIES    = ["B0005", "B0006", "B0007", "B0018"]
FEATURE_COLS = ["capacity", "IC_C_H", "IC_C_P", "IC_D_H", "IC_D_P"]
N_SYNTH      = 6000      # rows at start of combined file that are synthetic
WINDOW_SIZE  = 10        # sliding window size
N_AUG        = 30        # augmentation repetitions per training window
NOISE_STD    = 0.015     # Gaussian noise std for augmentation (in normalised space)
BATCH_SIZE   = 64
EPOCHS       = 400
LR           = 8e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 50
TRAIN_RATIO  = 0.80
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CASE2_COMBOS = [
    (["B0005", "B0006"], "B0007"),
    (["B0005", "B0006"], "B0018"),
    (["B0007", "B0006"], "B0005"),
    (["B0007", "B0018"], "B0006"),
]
print("Device:", DEVICE)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_data(battery: str) -> pd.DataFrame:
    """Extract all rows (synthetic + real), sorted by cycle."""
    path = os.path.join(HIS_DIR, "{}_synthetic+Real.csv".format(battery))
    df   = pd.read_csv(path)
    # The original strategy was to ignore synthetic data.
    # Now we include EVERYTHING (synthetic + real).
    df = df.sort_values("cycle").reset_index(drop=True)
    return df


def compute_rul(n: int) -> np.ndarray:
    return np.linspace(1.0, 0.0, n, dtype=np.float32)


# ── Augmented training dataset ────────────────────────────────────────────────

def build_augmented_windows(feats_norm: np.ndarray,
                             rul: np.ndarray) -> tuple:
    """
    Create sliding windows from normalised features, then replicate
    each window N_AUG times with small Gaussian noise added to features.
    Returns (X: Tensor, y: Tensor).
    """
    W = WINDOW_SIZE
    n = len(feats_norm)
    base_X, base_y = [], []
    for i in range(n - W + 1):
        base_X.append(feats_norm[i : i + W])
        base_y.append(rul[i + W - 1])

    if len(base_X) == 0:
        return torch.zeros(0, W, feats_norm.shape[1]), torch.zeros(0)

    base_X = np.array(base_X, dtype=np.float32)   # (N_windows, W, F)
    base_y = np.array(base_y, dtype=np.float32)   # (N_windows,)

    # Augment: repeat each window N_AUG times with noise
    aug_X = np.repeat(base_X, N_AUG, axis=0)        # (N*N_AUG, W, F)
    aug_y = np.repeat(base_y, N_AUG, axis=0)        # (N*N_AUG,)
    noise = np.random.normal(0, NOISE_STD, aug_X.shape).astype(np.float32)
    aug_X = np.clip(aug_X + noise, 0.0, 1.0)        # keep in [0,1]

    # Include original (clean) windows once more
    all_X = np.concatenate([base_X, aug_X], axis=0)
    all_y = np.concatenate([base_y, aug_y], axis=0)

    return torch.from_numpy(all_X), torch.from_numpy(all_y)


def make_test_loader(feats_norm: np.ndarray, rul: np.ndarray):
    W = WINDOW_SIZE
    n = len(feats_norm)
    X, y = [], []
    for i in range(n - W + 1):
        X.append(feats_norm[i : i + W])
        y.append(rul[i + W - 1])
    if len(X) == 0:
        return None
    X = torch.tensor(np.array(X, dtype=np.float32))
    y = torch.tensor(np.array(y, dtype=np.float32))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(X_train: torch.Tensor, y_train: torch.Tensor) -> SLMTransformerRUL:
    model     = SLMTransformerRUL(n_features=5, window_size=WINDOW_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    ds     = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    best_loss, best_state, no_imp = float("inf"), None, 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, ns = 0.0, 0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(yb)
            ns    += len(yb)
        scheduler.step()
        avg = total / max(ns, 1)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model

def evaluate(model, test_loader):
    if test_loader is None:
        return float("nan"), float("nan"), [], []
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            preds.extend(model(Xb.to(DEVICE)).cpu().numpy().tolist())
            targets.extend(yb.numpy().tolist())
    p, t = np.array(preds), np.array(targets)
    return mean_absolute_error(t, p), np.sqrt(mean_squared_error(t, p)), p, t


# ── Case-1 ────────────────────────────────────────────────────────────────────

def run_case1(battery: str, verbose=True):
    data  = load_all_data(battery)
    n     = len(data)
    rul   = compute_rul(n)
    feats = data[FEATURE_COLS].values

    # User specified split: 4935 for training, rest for testing.
    # Total data is 6168. 4935 + 1233 = 6168.
    split = 4935

    tr_feats, te_feats = feats[:split], feats[split:]
    tr_rul,   te_rul   = rul[:split],   rul[split:]

    scaler   = MinMaxScaler().fit(tr_feats)
    tr_norm  = scaler.transform(tr_feats)
    te_norm  = scaler.transform(te_feats)

    X_tr, y_tr = build_augmented_windows(tr_norm, tr_rul)
    test_loader = make_test_loader(te_norm, te_rul)

    n_aug_windows = len(X_tr)
    model = train_model(X_tr, y_tr)
    mae, rmse, preds, targets = evaluate(model, test_loader)

    if verbose:
        print("  Case-1 | {} | MAE: {:.4f} | RMSE: {:.4f}  "
              "[train_size={}, test_size={}, aug_windows={}]".format(
               battery, mae, rmse, split, n - split, n_aug_windows))
    return mae, rmse, model


# ── Case-2 ────────────────────────────────────────────────────────────────────

def run_case2(train_batteries: list, test_battery: str, verbose=True):
    train_datasets = [load_all_data(b) for b in train_batteries]
    test_dataset   = load_all_data(test_battery)

    all_tr_feats = np.vstack([d[FEATURE_COLS].values for d in train_datasets])
    scaler = MinMaxScaler().fit(all_tr_feats)

    all_X, all_y = [], []
    for d in train_datasets:
        n     = len(d)
        split = 4935  # Consistently use the same split point if applicable
        if n < split + WINDOW_SIZE:
             # Fallback for smaller datasets (though here they are expected to be 6168)
             split = int(n * 0.8)

        norm  = scaler.transform(d[FEATURE_COLS].values[:split])
        rul   = compute_rul(split)
        Xb, yb = build_augmented_windows(norm, rul)
        all_X.append(Xb);  all_y.append(yb)

    X_tr = torch.cat(all_X, dim=0)
    y_tr = torch.cat(all_y, dim=0)

    te_norm     = scaler.transform(test_dataset[FEATURE_COLS].values)
    te_rul      = compute_rul(len(test_dataset))
    test_loader = make_test_loader(te_norm, te_rul)

    model = train_model(X_tr, y_tr)
    mae, rmse, _, _ = evaluate(model, test_loader)
    if verbose:
        print("  Case-2 | Train={} -> Test={} | MAE: {:.4f} | RMSE: {:.4f}".format(
              "+".join(train_batteries), test_battery, mae, rmse))
    return mae, rmse, model


# ── Case-3 ────────────────────────────────────────────────────────────────────

def run_case3(held_out: str, verbose=True):
    train_batteries = [b for b in BATTERIES if b != held_out]
    mae, rmse, model = run_case2(train_batteries, held_out, verbose=False)
    if verbose:
        print("  Case-3 | Train={} -> Test={} | MAE: {:.4f} | RMSE: {:.4f}".format(
              "+".join(train_batteries), held_out, mae, rmse))
    return mae, rmse, model


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case",    type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--battery", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("SLM RUL Training  --  Case {}".format(args.case))
    print("=" * 60)

    if args.case == 1:
        targets = [args.battery] if args.battery else BATTERIES
        results = {}
        for b in targets:
            mae, rmse, _ = run_case1(b)
            results[b] = (mae, rmse)

        maes  = [v[0] for v in results.values()]
        rmses = [v[1] for v in results.values()]
        print("\nSummary (Case-1):")
        print("{:<8} | {:>10} | {:>10}".format("Battery", "MAE", "RMSE"))
        print("-" * 35)
        for b, (mae, rmse) in results.items():
            print("{:<8} | {:>10.4f} | {:>10.4f}".format(b, mae, rmse))
        if len(maes) > 1:
            print("-" * 35)
            print("{:<8} | {:>10.4f} | {:>10.4f}".format(
                  "AVG", np.mean(maes), np.mean(rmses)))

    elif args.case == 2:
        for tb, testb in CASE2_COMBOS:
            run_case2(tb, testb)

    elif args.case == 3:
        for b in BATTERIES:
            run_case3(b)


if __name__ == "__main__":
    main()
