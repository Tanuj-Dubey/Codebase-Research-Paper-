import pandas as pd
import numpy as np
import os
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler

# Configuration
base_dir = r"d:\research paper - Copy\code+data\code+data\dataset\csv_format"
batteries = ["B0005", "B0006", "B0007", "B0018"]
synthetic_file = "synthetic_battery_cycles_50k_strict.csv"

def calculate_kl_divergence(real_df, synth_df):
    results = {}
    columns = real_df.columns
    
    for col in columns:
        if col not in synth_df.columns:
            print(f"Warning: Column {col} not found in synthetic data.")
            continue
            
        real_data = real_df[col].dropna().values
        synth_data = synth_df[col].dropna().values
        
        # Scale to [0,1] based on real data for consistent binning
        scaler = MinMaxScaler()
        real_scaled = scaler.fit_transform(real_data.reshape(-1, 1)).flatten()
        synth_scaled = scaler.transform(synth_data.reshape(-1, 1)).flatten()
        
        # Compute histograms to get counts (not density) for Laplace smoothing
        bins = 50
        hist_real, bin_edges = np.histogram(real_scaled, bins=bins, range=(0, 1))
        hist_synth, _ = np.histogram(synth_scaled, bins=bin_edges)
        
        # Laplace Smoothing: Add 1 to each bin to avoid zero probabilities
        # This is a standard technique for KL divergence on discrete distributions
        p = (hist_real.astype(float) + 1.0)
        q = (hist_synth.astype(float) + 1.0)
        
        # Normalize to PMF
        p /= p.sum()
        q /= q.sum()
        
        # Calculate KL Divergence D_KL(P || Q)
        kl_div = entropy(p, q)
        results[col] = kl_div
        
    return results

def main():
    print("--- Loading and Aggregating Real NASA Battery Data ---")
    cycle_data = []
    for b in batteries:
        path = os.path.join(base_dir, f"{b}_discharge.csv")
        if not os.path.exists(path):
            print(f"Error: {path} not found.")
            continue
            
        df = pd.read_csv(path)
        # Aggregation logic matched with VAE training scripts
        agg_df = df.groupby('id_cycle').agg({
            'Voltage_measured': ['max', 'min', 'mean'],
            'Current_measured': ['max', 'min', 'mean'],
            'Temperature_measured': 'mean',
            'Capacity': 'max'
        })
        # Flatten multi-index columns
        agg_df.columns = [f"{c[0]}_{c[1]}" for c in agg_df.columns]
        cycle_data.append(agg_df)
        print(f"  Processed {b} discharge cycles.")

    if not cycle_data:
        print("No real data loaded. Exiting.")
        return
        
    real_df = pd.concat(cycle_data, ignore_index=True)
    
    print(f"\n--- Loading Synthetic Data: {synthetic_file} ---")
    synth_path = os.path.join(base_dir, synthetic_file)
    if not os.path.exists(synth_path):
        print(f"Error: {synth_path} not found.")
        return
        
    synth_df = pd.read_csv(synth_path)
    
    print("\n--- Performing KL Divergence Test ---")
    kl_results = calculate_kl_divergence(real_df, synth_df)
    
    print("\nResults (D_KL(Real || Synthetic)):")
    print("-" * 40)
    for col, kl in kl_results.items():
        print(f"{col:<30}: {kl:.6f}")
    
    avg_kl = np.mean(list(kl_results.values()))
    print("-" * 40)
    print(f"{'Average KL Divergence':<30}: {avg_kl:.6f}")
    print("\nInterpretation: Lower values indicate higher similarity between distributions.")

if __name__ == "__main__":
    main()
