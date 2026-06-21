import matplotlib.pyplot as plt
import os

# --- Comprehensive Configuration Data ---
# Group 1: Data Protocol and Splitting
protocol_params = [
    ["Parameter", "Value", "Description"],
    ["Observations (N)", "3,000", "Chronological cycles retained per battery"],
    ["Sequence Length", "96", "Input history for Base Forecasting"],
    ["Train/Test Split", "80/20", "Case 1 temporal data allocation"],
    ["LOO Portions", "3 Train / 1 Test", "Case 3 Cross-battery fold design"],
    ["Scalers", "Standard / MinMax", "Strict train-only fitment protocol"]
]

# Group 2: Generative SLM Training (Individual Case)
slm_params = [
    ["Parameter", "Value", "Description"],
    ["Window Size (W)", "10", "Localized context for SLM input"],
    ["Augmentation (N_AUG)", "30x", "Synthetic samples per training window"],
    ["Noise Std", "0.015", "Gaussian deviation for feature synthesis"],
    ["Epochs / Patience", "400 / 50", "Training cycles with early stopping"],
    ["Optimizer (AdamW)", "8e-4", "Learning rate with Cosine Annealing"],
    ["Weight Decay", "1e-4", "L2 regularization constant"]
]

# Group 3: Hybrid & Residual Gating (Advanced)
hybrid_params = [
    ["Parameter", "Value", "Description"],
    ["Chronos Samples", "32", "Stochastic forecasting paths"],
    ["Context Spans", "(96, 64, 48)", "Multi-contextual trailing history"],
    ["Residual Scale", "10.0", "Error-to-target multiplier"],
    ["Gate Alpha (\u03B1)", "0.85", "Residual modulation strength"],
    ["EMA Beta (\u03B2)", "0.70", "Causal smoothing across batches"],
    ["Gate [Floor, Cap]", "[0.2, 1.0]", "Adaptive confidence clipping"]
]

# Group 4: Architecture Specifications
architecture_params = [
    ["Model", "Dimensions", "Key Specifications"],
    ["SLM Transformer", "64", "2 Layers, 4 Heads, d_ff=128"],
    ["PatchTST Residual", "96", "Patch=16, Stride=8, 2 Layers"],
    ["TinyTimeMixer", "96", "MLP-based residual projector"],
    ["Base Forecaster", "Base", "Amazon Chronos T5 (54M params)"],
    ["Calibration", "Ridge", "Affine map with \u03BB=1e-3"]
]

def create_comprehensive_image():
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    plt.subplots_adjust(hspace=0.25)
    
    datasets = [protocol_params, slm_params, hybrid_params, architecture_params]
    titles = [
        "Table 5.1: Data Protocol & Generalization Design",
        "Table 5.2: Generative SLM Training (Intra-Battery Case)",
        "Table 5.3: Hybrid Forecasting & Advanced Gating Parameters",
        "Table 5.4: Layer-wise Architecture & Calibration Detail"
    ]
        
    for i, (data, title) in enumerate(zip(datasets, titles)):
        ax = axes[i]
        ax.axis('off')
        
        table = ax.table(cellText=data, loc='center', cellLoc='left', colWidths=[0.25, 0.20, 0.45])
        table.auto_set_font_size(False)
        table.set_fontsize(10.5)
        table.scale(1.1, 2.4)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#1F618D') # Academic dark blue
            else:
                if row % 2 == 0:
                    cell.set_facecolor('#FDFEFE')
                else:
                    cell.set_facecolor('#F2F4F4')
            cell.set_edgecolor('#D5DBDB')
            
        ax.set_title(title, fontsize=15, fontweight='bold', color='#154360', pad=12)

    os.makedirs('plots', exist_ok=True)
    out_path = "plots/comprehensive_hyperparameters.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    return out_path

new_path = create_comprehensive_image()
print(f"Verified and saved: {new_path}")
