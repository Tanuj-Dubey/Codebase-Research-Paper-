import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

HIS_DIR = r"code+data/code+data/dataset/HIS"
OUTPUT_DIR = r"code+data/code+data/dataset/plots"
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
FEATURE_COLS = ["capacity", "IC_C_H", "IC_C_P", "IC_D_H", "IC_D_P"]
N_SYNTH = 6000

os.makedirs(OUTPUT_DIR, exist_ok=True)

for battery in BATTERIES:
    print(f"Generating t-SNE for {battery}...")
    file_path = os.path.join(HIS_DIR, f"{battery}_synthetic+Real.csv")
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    df = pd.read_csv(file_path)
    
    # Identify Synthetic vs Real: first N_SYNTH are synthetic
    labels = np.array(['Synthetic'] * N_SYNTH + ['Real'] * (len(df) - N_SYNTH))
    features = df[FEATURE_COLS].values
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Run t-SNE
    # Using a subset if needed, but 6000+ points is manageable for TSNE (might take ~30-60s)
    # n_jobs=-1 speeds it up if supported.
    tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    # Plot Synthetic first so Real (which is fewer points) is plotted on top
    synth_mask = (labels == 'Synthetic')
    real_mask = (labels == 'Real')
    
    plt.scatter(features_tsne[synth_mask, 0], features_tsne[synth_mask, 1], 
                alpha=0.3, label='Synthetic', color='cyan', edgecolors='none', s=10)
    plt.scatter(features_tsne[real_mask, 0], features_tsne[real_mask, 1], 
                alpha=0.9, label='Real Data', color='red', edgecolors='black', s=30)
    
    plt.title(f"{battery} - t-SNE Projection (Synthetic vs Real)")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(OUTPUT_DIR, f"{battery}_tsne.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved {out_path}")

print("All t-SNE plots generated successfully!")
