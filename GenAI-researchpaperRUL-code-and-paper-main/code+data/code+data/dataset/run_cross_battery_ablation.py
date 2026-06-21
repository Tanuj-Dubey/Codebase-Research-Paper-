import os
import itertools
import pandas as pd
import numpy as np
import torch
import gc
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from rich.console import Console

# Set environment variables for fixed parameters BEFORE importing pipeline
os.environ["BATCH_SIZE"] = "128"
os.environ["LEARNING_RATE"] = "0.01"
os.environ["EPOCHS"] = "50"
os.environ["DROPOUT"] = "0.1"
os.environ["HYBRID_MODE"] = "1"
os.environ["CHRONOS_QUIET"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import hybrid_pipeline as hp

CONSOLE = Console()

# --- Custom Data Loader for Cross-Battery ---
def get_cross_battery_data(train_batteries, test_battery):
    train_dfs = []
    for b in train_batteries:
        p = os.path.join(hp.HIS_DIR, f"{b}_synthetic+Real.csv")
        df = pd.read_csv(p).sort_values("cycle").reset_index(drop=True)
        if len(df) > hp.LAST_N_ROWS:
            df = df.tail(hp.LAST_N_ROWS).reset_index(drop=True)
        train_dfs.append(df)
        
    p_test = os.path.join(hp.HIS_DIR, f"{test_battery}_synthetic+Real.csv")
    df_test = pd.read_csv(p_test).sort_values("cycle").reset_index(drop=True)
    if len(df_test) > hp.LAST_N_ROWS:
        df_test = df_test.tail(hp.LAST_N_ROWS).reset_index(drop=True)

    feat_cols = [c for c in df_test.columns if c != "cycle"]
    cap_idx = feat_cols.index("capacity")
    n_feat = len(feat_cols)
    
    scaler = StandardScaler()
    full_train_df = pd.concat(train_dfs, ignore_index=True)
    scaler.fit(full_train_df[feat_cols].values.astype(np.float32))
    
    test_sc = scaler.transform(df_test[feat_cols].values.astype(np.float32))
    
    def window(arr):
        if len(arr) <= hp.SEQ_LEN:
            return (
                np.empty((0, hp.SEQ_LEN, n_feat), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        v = np.lib.stride_tricks.sliding_window_view(arr, hp.SEQ_LEN, axis=0)
        v = v.transpose(0, 2, 1)
        X = v[:-1]
        y = arr[hp.SEQ_LEN:, cap_idx].reshape(-1, 1)
        lc = v[:-1, -1, cap_idx].reshape(-1, 1)
        return X, y, lc

    X_trains, y_trains, lc_trains = [], [], []
    for df in train_dfs:
        sc = scaler.transform(df[feat_cols].values.astype(np.float32))
        X, y, lc = window(sc)
        X_trains.append(X); y_trains.append(y); lc_trains.append(lc)
        
    X_train = np.concatenate(X_trains)
    y_train = np.concatenate(y_trains)
    lc_train = np.concatenate(lc_trains)
    
    X_test, y_test, lc_test = window(test_sc)
    
    return X_train, y_train, lc_train, X_test, y_test, lc_test, cap_idx, scaler

# Override Chronos cache path
original_cache_path = hp._chronos_cache_path
def custom_cache_path(split_name, context_len, battery=None):
    if battery and "cross_" in battery:
        return os.path.join(hp.HIS_DIR, f"chronos_{hp.CHRONOS_CACHE_TAG}_{battery}_{split_name}_L{context_len}.npy")
    return original_cache_path(split_name, context_len, battery)
hp._chronos_cache_path = custom_cache_path

# --- Execution Function ---
def run_cross_battery_hybrid(train_batteries, test_battery, pipeline, model_used):
    train_str = "".join(b[-1] for b in train_batteries)
    test_str = test_battery[-1]
    cross_bat_id = f"cross_Tr{train_str}_Te{test_str}"
    
    X_tr, y_tr, lc_tr, X_te, y_te, lc_te, cap_idx, scaler = get_cross_battery_data(train_batteries, test_battery)
    
    c_tr = hp.get_split_chronos(X_tr, cap_idx, "train", battery=cross_bat_id)
    c_te = hp.get_split_chronos(X_te, cap_idx, "test", battery=cross_bat_id)
    
    # We use 85/15 split of the training data just for early stopping validation
    cut = max(1, int(len(X_tr) * 0.85))
    X_tr_m, X_val_m = X_tr[:cut], X_tr[cut:]
    y_tr_m, y_val_m = y_tr[:cut], y_tr[cut:]
    lc_tr_m, lc_val_m = lc_tr[:cut], lc_tr[cut:]
    c_tr_m, c_val_m = c_tr[:cut], c_tr[cut:]
    
    w_tr = torch.linspace(1.0, 2.0, len(X_tr_m)).reshape(-1, 1)
    
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_m), c_tr_m, torch.from_numpy(lc_tr_m), torch.from_numpy(y_tr_m), w_tr), batch_size=hp.BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_val_m), c_val_m, torch.from_numpy(lc_val_m), torch.from_numpy(y_val_m)), batch_size=hp.BATCH_SIZE)
    te_loader = DataLoader(TensorDataset(torch.from_numpy(X_te), c_te, torch.from_numpy(lc_te), torch.from_numpy(y_te)), batch_size=hp.BATCH_SIZE)
    
    m1 = hp.PatchTSTResidual(X_tr.shape[2], hp.SEQ_LEN, dropout=0.1)
    m2 = hp.TinyTimeMixer(X_tr.shape[2], hp.SEQ_LEN, dropout=0.1)
    
    hp.train_model(m1, tr_loader, val_loader, epochs=50, threshold=None, lr=0.01)
    hp.train_model(m2, tr_loader, val_loader, epochs=50, threshold=None, lr=0.01)
    
    m1.load_state_dict(torch.load("best_PatchTSTResidual.pth"))
    m2.load_state_dict(torch.load("best_TinyTimeMixer.pth"))
    m1.eval(); m2.eval(); m1.to(hp.DEVICE); m2.to(hp.DEVICE)
    
    te_preds, te_true, te_base = [], [], []
    with torch.no_grad():
        for Xb, cpb, lcb, yb in te_loader:
            Xb, cpb, lcb, yb = Xb.to(hp.DEVICE), cpb.to(hp.DEVICE), lcb.to(hp.DEVICE), yb.to(hp.DEVICE)
            p1 = (cpb.cpu().numpy() if cpb.is_cuda else cpb.numpy()).reshape(-1)
            r1 = (m1(Xb, cpb, lcb) / 10.0).cpu().numpy().flatten()
            r2 = (m2(Xb, cpb, lcb) / 10.0).cpu().numpy().flatten()
            te_preds.append(0.5 * (p1 + r1) + 0.5 * (p1 + r2))
            te_true.append(yb.cpu().numpy().flatten())
            te_base.append(p1)
            
    p_te, y_te_f, b_te = np.concatenate(te_preds), np.concatenate(te_true), np.concatenate(te_base)
    std, mean = scaler.scale_[cap_idx], scaler.mean_[cap_idx]
    
    p_un, y_un, b_un = p_te * std + mean, y_te_f * std + mean, b_te * std + mean
    
    mae = mean_absolute_error(y_un, p_un)
    rmse = np.sqrt(mean_squared_error(y_un, p_un))
    r2 = r2_score(y_un, p_un)
    
    return mae, rmse, r2

# --- Main Ablation Loop ---
batteries = ["B0005", "B0006", "B0007", "B0018"]
permutations = []

# Single to Single
for train_b in batteries:
    for test_b in batteries:
        if train_b != test_b:
            permutations.append(([train_b], test_b, "1 to 1"))

# 2 to Single
for train_pair in itertools.combinations(batteries, 2):
    for test_b in batteries:
        if test_b not in train_pair:
            permutations.append((list(train_pair), test_b, "2 to 1"))

# 3 to Single (LOO)
for train_trio in itertools.combinations(batteries, 3):
    for test_b in batteries:
        if test_b not in train_trio:
            permutations.append((list(train_trio), test_b, "3 to 1 (LOO)"))

RESULTS_CSV = "cross_battery_ablation.csv"
if os.path.exists(RESULTS_CSV):
    existing_df = pd.read_csv(RESULTS_CSV)
    done_set = set(zip(existing_df['Train'], existing_df['Test']))
else:
    existing_df = pd.DataFrame()
    done_set = set()

results = []
print(f"Total Combinations to run: {len(permutations)}")

pipeline, model_used = hp._load_chronos_pipeline()

for i, (train_bs, test_b, exp_type) in enumerate(permutations):
    train_str = "+".join(train_bs)
    if (train_str, test_b) in done_set:
        print(f"[{i+1}/{len(permutations)}] Skipping {train_str} -> {test_b} (Already Done)")
        continue
        
    print(f"\n[{i+1}/{len(permutations)}] Running: Train={train_str} | Test={test_b} | Type={exp_type}")
    try:
        mae, rmse, r2 = run_cross_battery_hybrid(train_bs, test_b, pipeline, model_used)
        results.append({
            "Type": exp_type,
            "Train": train_str,
            "Test": test_b,
            "R2": r2,
            "RMSE": rmse,
            "MAE": mae
        })
        pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
        print(f"Success! MAE: {mae:.6f} | R2: {r2:.6f}")
    except Exception as e:
        print(f"Error on {train_str} -> {test_b}: {e}")

if not existing_df.empty:
    final_df = pd.concat([existing_df, pd.DataFrame(results)], ignore_index=True)
else:
    final_df = pd.DataFrame(results)
    
final_df.to_csv(RESULTS_CSV, index=False)
print("\n--- ABLATION STUDY COMPLETE ---")
