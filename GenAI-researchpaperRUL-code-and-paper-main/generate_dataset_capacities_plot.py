import csv
import matplotlib.pyplot as plt
import os
import numpy as np

# Configure aesthetic styles
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'bold'
})

def main():
    # Base path settings
    base_dir = r"c:\Users\AADITYA COM\OneDrive\Desktop\GenAI-researchpaperRUL-code-and-paper-main\code+data\code+data\dataset"
    his_dir = os.path.join(base_dir, "HIS")
    plots_dir = os.path.join(base_dir, "plots")
    
    os.makedirs(plots_dir, exist_ok=True)
    
    batteries = ["B0005", "B0006", "B0007", "B0018"]
    colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'] # Distinct aesthetic colors
    
    # 2x2 Grid for the 4 batteries
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Battery Capacity Degradation Over Cycles', fontweight='bold', fontsize=18, y=0.98)
    fig.patch.set_facecolor('#F8F9FA')
    axs = axs.flatten()
    
    for idx, (bat, color) in enumerate(zip(batteries, colors)):
        csv_path = os.path.join(his_dir, f"{bat}_synthetic+Real.csv")
        ax = axs[idx]
        ax.set_facecolor('#FFFFFF')
        
        if os.path.exists(csv_path):
            cycles = []
            capacities = []
            
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        c = float(row['cycle'])
                        cap = float(row['capacity'])
                        cycles.append(c)
                        capacities.append(cap)
                    except ValueError:
                        pass
            
            # Scatter and Line plot
            ax.plot(cycles, capacities, color=color, linewidth=2, alpha=0.8, label=f'{bat} Capacity')
            ax.scatter(cycles, capacities, color=color, s=10, alpha=0.5)
            
            # Annotations and Labels
            ax.set_title(f'Battery {bat}', color='#333333', pad=10)
            ax.set_xlabel('Cycle Number')
            ax.set_ylabel('Capacity (Ah)')
            
            # Clean grid lines
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='upper right')
            
            if cycles:
                # Highlighting the initial and final capacity values
                init_cap = capacities[0]
                final_cap = capacities[-1]
                
                ax.annotate(f"Init: {init_cap:.2f}Ah", 
                            (cycles[0], init_cap), 
                            textcoords="offset points", xytext=(10,-10), ha='left', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1, alpha=0.8))
                            
                ax.annotate(f"Final: {final_cap:.2f}Ah", 
                            (cycles[-1], final_cap), 
                            textcoords="offset points", xytext=(-10,10), ha='right', fontsize=9,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1, alpha=0.8))
        else:
            ax.text(0.5, 0.5, f"{bat} Data Not Found", ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f'Battery {bat}')
            
    plt.tight_layout()
    fig.subplots_adjust(top=0.9) # Adjust top to accommodate the main title
    
    # Save specifically in the plot folder to store in a single place
    output_path = os.path.join(plots_dir, "combined_battery_capacities.png")
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"SUCCESS: {output_path}")

if __name__ == "__main__":
    main()
