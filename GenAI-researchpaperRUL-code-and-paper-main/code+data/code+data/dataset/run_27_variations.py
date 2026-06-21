import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# --- CONFIGURATION ---
BATTERY = "B0005"
BASE_MAE = 0.0097
BASE_RMSE = 0.0127

# Define the 27 variations (3 Batch Sizes x 3 Learning Rates x 3 Epochs)
# We use Epochs 30, 50, 100 to include the "higher epoch" requested
batch_sizes = [32, 64, 128]
learning_rates = [0.0001, 0.001, 0.01]
epochs_list = [30, 50, 100]  # Includes higher epoch 100

experiments = []
for bs in batch_sizes:
    dr = 0.02 if bs == 32 else (0.05 if bs == 64 else 0.1)
    for ep in epochs_list:
        for lr in learning_rates:
            experiments.append({"BS": bs, "LR": lr, "EP": ep, "DR": dr})

results_csv = "sensitivity_results_27.csv"

# Load existing results if any
if os.path.exists(results_csv):
    existing_df = pd.read_csv(results_csv)
    existing_configs = set(zip(existing_df['Batch size'], existing_df['Learning rate'], existing_df['Epoch']))
else:
    existing_df = pd.DataFrame()
    existing_configs = set()

print(f"Starting execution of 27 variations for Battery {BATTERY}...")
print(f"Epochs: {epochs_list} | LRs: {learning_rates} | BS: {batch_sizes}")
print("-" * 80)

all_results = []

for i, exp in enumerate(experiments):
    bs, lr, ep, dr = exp["BS"], exp["LR"], exp["EP"], exp["DR"]
    
    if (bs, lr, ep) in existing_configs:
        print(f"Exp {i+1}/27: BS={bs}, LR={lr}, Ep={ep} [ALREADY DONE]")
        # Keep existing row
        row = existing_df[(existing_df['Batch size'] == bs) & 
                          (existing_df['Learning rate'] == lr) & 
                          (existing_df['Epoch'] == ep)].iloc[0].to_dict()
        all_results.append(row)
        continue

    print(f"Exp {i+1}/27: BS={bs}, LR={lr}, Ep={ep}, Dropout={dr} [RUNNING...]")
    
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(bs)
    env["LEARNING_RATE"] = str(lr)
    env["EPOCHS"] = str(ep)
    env["DROPOUT"] = str(dr)
    env["CHRONOS_QUIET"] = "1"
    env["HYBRID_MODE"] = "1"
    
    try:
        t0 = time.time()
        subprocess.run(["python", "hybrid_pipeline.py", BATTERY], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        
        res_path = f"HIS/final_SOTA_{BATTERY}.csv"
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            mae = mean_absolute_error(df['true'], df['pred'])
            rmse = np.sqrt(mean_squared_error(df['true'], df['pred']))
            r2 = r2_score(df['true'], df['pred'])
            mae_imp = ((BASE_MAE - mae) / BASE_MAE) * 100
            
            result = {
                "Case": i + 1,
                "Batch size": bs,
                "Learning rate": lr,
                "Epoch": ep,
                "dropout": dr,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAE Imp%": f"{mae_imp:.2f}%"
            }
            all_results.append(result)
            print(f"  Done in {dt:.1f}s | MAE: {mae:.6f} ({mae_imp:+.2f}%)")
            
            # Save intermediate results
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
        else:
            print(f"  Error: Results file {res_path} not found.")
    except Exception as e:
        print(f"  Failed: {e}")

# Final Save
final_df = pd.DataFrame(all_results)
final_df.to_csv("sensitivity_results_27.csv", index=False)
final_df.to_excel("sensitivity_results_27.xlsx", index=False)

print("\nAll 27 variations completed!")
print("-" * 80)
print(final_df.to_string(index=False))
