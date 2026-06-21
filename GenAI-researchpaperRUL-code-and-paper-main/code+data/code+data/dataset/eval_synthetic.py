import pandas as pd
import numpy as np
import os
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
synthetic_file = os.path.join(base_dir, "synthetic_battery_cycles_50k_strict.csv")
batteries = ["B0005", "B0006", "B0007", "B0018"]

# 1. Load Real Data
print("Loading real dataset...")
cycle_data = []
for b in batteries:
    df = pd.read_csv(os.path.join(base_dir, f"{b}_discharge.csv"))
    agg_df = df.groupby('id_cycle').agg({
        'Voltage_measured': ['max', 'min', 'mean'],
        'Current_measured': ['max', 'min', 'mean'],
        'Temperature_measured': 'mean',
        'Capacity': 'max'
    })
    agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
    cycle_data.append(agg_df)

real_df = pd.concat(cycle_data, ignore_index=True)

# 2. Load Synthetic Data
print("Loading synthetic dataset...")
synth_df = pd.read_csv(synthetic_file)

# We must ensure columns match identically
columns_to_compare = synth_df.columns.tolist()

# 3. Calculate Divergence Metrics per feature
print("\n--- Statistical Divergence Metrics ---")

def jensen_shannon_divergence(P, Q):
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

total_wd = 0
total_js = 0

print(f"{'Feature':<30} | {'Wasserstein Dist':<18} | {'JS Divergence':<15}")
print("-" * 68)

for col in columns_to_compare:
    # Get arrays
    real_data = real_df[col].dropna().values
    synth_data = synth_df[col].dropna().values
    
    # Normalize features to [0,1] range so Wasserstein distance is comparable across features
    scaler = MinMaxScaler()
    # Fit scaler on real data, transform both real and synthetic
    real_scaled = scaler.fit_transform(real_data.reshape(-1, 1)).flatten()
    synth_scaled = scaler.transform(synth_data.reshape(-1, 1)).flatten()
    
    # 1. Wasserstein Distance (Earth Mover's Distance)
    wd = wasserstein_distance(real_scaled, synth_scaled)
    total_wd += wd
    
    # 2. Jensen-Shannon Divergence
    # We need to bin the continuous data into discrete probability distributions for JS/KL
    hist_real, bin_edges = np.histogram(real_scaled, bins=50, range=(0,1), density=True)
    hist_synth, _ = np.histogram(synth_scaled, bins=bin_edges, density=True)
    
    # Add epsilon to avoid log(0)
    eps = 1e-10
    hist_real = hist_real + eps
    hist_synth = hist_synth + eps
    
    # JS Divergence is symmetric, bounded [0, 1] (or [0, ln(2)])
    js = jensen_shannon_divergence(hist_real, hist_synth)
    total_js += js
    
    print(f"{col:<30} | {wd:<18.4f} | {js:<15.4f}")

print("-" * 68)
avg_wd = total_wd / len(columns_to_compare)
avg_js = total_js / len(columns_to_compare)

print(f"\nAverage Normalized Wasserstein Distance: {avg_wd:.4f}")
print(f"Average Jensen-Shannon Divergence: {avg_js:.4f}")

if avg_js < 0.2:
    print("-> Assessment: High morphological similarity.")
elif avg_js < 0.5:
    print("-> Assessment: Moderate morphological similarity.")
else:
    print("-> Assessment: Low morphological similarity. Synthetic data distribution diverges significantly from real data.")
