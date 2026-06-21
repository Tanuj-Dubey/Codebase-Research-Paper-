import matplotlib.pyplot as plt
import numpy as np
import os

methods = ['AutoML 2022', 'Informer+HI', 'AutoML 2024 (H2O)', 'Chronos (No GAN)', 'Chronos + GAN']
mae = [0.0336, 0.0254, 0.0138, 0.0272, 0.0066]
rmse = [0.0414, 0.0331, 0.0196, 0.0370, 0.0130]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 7))

# Attractive colors: Teal and Coral
color_mae = '#1F77B4' 
color_rmse = '#FF7F0E'

rects1 = ax.bar(x - width/2, mae, width, label='MAE (Lower is Better)', color=color_mae, edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, rmse, width, label='RMSE (Lower is Better)', color=color_rmse, edgecolor='black', alpha=0.85)

ax.set_ylabel('Error Values', fontsize=14, fontweight='bold', labelpad=15)
ax.set_title('Progress in Battery Capacity Forecasting (Case-3: Leave-One-Out)', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=20, ha='right', fontsize=12, fontweight='medium')
ax.legend(fontsize=12, loc='upper right')

# Displaying rounded values on top
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# Highlight the final method (Chronos + GAN) manually by adding a background span
ax.axvspan(3.5, 4.5, color='#E8F5E9', alpha=0.5, zorder=0)

ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
ax.set_axisbelow(True)

ax.set_facecolor('#F8F9FA')
fig.patch.set_facecolor('#FFFFFF')

plt.tight_layout()

# Save to conversation artifact directory directly
output_path = r"C:\Users\AADITYA COM\.gemini\antigravity\brain\5d7fd841-bf29-440c-921d-6479826aca56\scratch\case3_progress.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS")
