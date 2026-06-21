"""
generate_his_synthetic.py
=========================
Generates 6,000 synthetic Health Index (HIS) rows per battery,
then combines with the real ~169 rows.

Output files (saved to the HIS folder):
  B0005_synthetic+Real.csv
  B0006_synthetic+Real.csv
  B0007_synthetic+Real.csv
  B0018_synthetic+Real.csv

Each file schema:  cycle, capacity, IC_C_H, IC_C_P, IC_D_H, IC_D_P
"""

import os
import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
HIS_DIR     = r"d:\research paper - Copy\code+data\code+data\dataset\HIS"
BATTERIES   = ["B0005", "B0006", "B0007", "B0018"]
N_SYNTH     = 6000
RANDOM_SEED = 42

# Columns that are log-transformed before Gaussian fitting
# (heavy-tailed positive distributions)
LOG_COLS = ["IC_C_H", "IC_D_H"]

np.random.seed(RANDOM_SEED)


# ── Helpers ──────────────────────────────────────────────────────────────────

def safe_log(arr):
    """Log-transform with safeguard for non-positive values."""
    return np.log(np.clip(arr, 1e-6, None))


def safe_exp(arr):
    return np.exp(arr)


def fit_and_sample_cycle_aware(real_df: pd.DataFrame, n_total: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Generate synthetic samples by iterating through real cycles and adding 
    controlled noise to each. This preserves the temporal degradation signal.
    """
    feature_cols = ["cycle", "capacity", "IC_C_H", "IC_C_P", "IC_D_H", "IC_D_P"]
    real_data = real_df[feature_cols].copy()
    n_real = len(real_data)
    
    # Calculate samples per real cycle
    samples_per_cycle = n_total // n_real
    remainder = n_total % n_real
    
    synth_list = []
    
    # Define noise levels (percentage of feature value)
    # Different features might need different noise scales
    noise_scales = {
        "capacity": 0.002,   # 0.2% noise
        "IC_C_H":   0.05,    # 5% noise
        "IC_C_P":   0.02,    # 2% noise
        "IC_D_H":   0.05,    # 5% noise
        "IC_D_P":   0.02,    # 2% noise
    }

    for i, row in real_data.iterrows():
        n_to_gen = samples_per_cycle + (1 if i < remainder else 0)
        
        for _ in range(n_to_gen):
            s_row = row.copy()
            for feat, scale in noise_scales.items():
                # Add Gaussian noise
                noise = rng.normal(0, abs(row[feat]) * scale)
                s_row[feat] += noise
            
            synth_list.append(s_row)
            
    synth_df = pd.DataFrame(synth_list)
    
    # Clip to realistic [min, max] of real data to avoid outliers
    for col in feature_cols:
        if col == "cycle": continue
        lo = real_df[col].min() * 0.95
        hi = real_df[col].max() * 1.05
        synth_df[col] = synth_df[col].clip(lo, hi)
        
    return synth_df.reset_index(drop=True)


def main():
    print("=" * 60)
    print("HIS Synthetic Data Generator (CYCLE-AWARE VERSION)")
    print("=" * 60)
    rng = np.random.default_rng(RANDOM_SEED)

    if not os.path.exists(HIS_DIR):
        os.makedirs(HIS_DIR)

    for battery in BATTERIES:
        print(f"\nProcessing {battery}...")
        input_path  = os.path.join(HIS_DIR, f"{battery}_health_index_updated.csv")
        output_path = os.path.join(HIS_DIR, f"{battery}_synthetic+Real.csv")

        if not os.path.exists(input_path):
            print(f"  [SKIP] {input_path} not found.")
            continue

        real_df = pd.read_csv(input_path)
        real_df = real_df.dropna(subset=["cycle", "capacity"]).reset_index(drop=True)
        n_real = len(real_df)

        # ── Generate cycle-aware synthetic features
        synth_df = fit_and_sample_cycle_aware(real_df, N_SYNTH, rng)
        
        # ── Combined (Synthetic first, then Real - for test_his_quality.py)
        combined = pd.concat([synth_df, real_df], ignore_index=True)

        combined.to_csv(output_path, index=False)
        print(f"  {battery}: saved {len(combined)} rows -> {output_path}")
        print(f"           (synthetic={len(synth_df)}, real={n_real})")

    print("\nAll batteries processed successfully.")
    print(f"\nOutput files in: {HIS_DIR}")


if __name__ == "__main__":
    main()
