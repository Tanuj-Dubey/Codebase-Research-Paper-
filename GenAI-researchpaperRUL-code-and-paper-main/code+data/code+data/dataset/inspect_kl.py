import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
batteries = ["B0005", "B0006", "B0007", "B0018"]
synthetic_file = os.path.join(base_dir, "synthetic_battery_cycles_50k_strict.csv")

# Load real data
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
synth_df = pd.read_csv(synthetic_file)

col = 'Voltage_measured_max'
real_data = real_df[col].dropna().values
synth_data = synth_df[col].dropna().values

scaler = MinMaxScaler()
real_scaled = scaler.fit_transform(real_data.reshape(-1, 1)).flatten()
synth_scaled = scaler.transform(synth_data.reshape(-1, 1)).flatten()

bins = 50
hist_real, bin_edges = np.histogram(real_scaled, bins=bins, range=(0, 1), density=True)
hist_synth, _ = np.histogram(synth_scaled, bins=bin_edges, density=True)

eps = 1e-10
p = (hist_real + eps)
q = (hist_synth + eps)
p /= p.sum()
q /= q.sum()

kl_div = entropy(p, q)
print(f"KL Divergence for {col}: {kl_div}")

# Analysis
print(f"Number of bins where Real > 0 but Synthetic is tiny: {np.sum((hist_real > 1e-5) & (hist_synth < 1e-5))}")
print(f"Contribution of 'mismatched' bins to KL divergence:")
kl_per_bin = p * np.log(p/q)
print(f"Sum of top 5 KL contributions: {np.sum(np.sort(kl_per_bin)[-5:])}")
print(f"Total KL sum: {np.sum(kl_per_bin)}")
