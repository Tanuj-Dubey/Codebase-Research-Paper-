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

# Define the full 36-variation grid (3 BS x 3 LR x 4 Epochs: 20, 30, 50, 100)
batch_sizes = [32, 64, 128]
learning_rates = [0.0001, 0.001, 0.01]
epochs_list = [20, 30, 50, 100]  # Includes the existing ones + the higher 100 epoch

experiments = []
for bs in batch_sizes:
    dr = 0.02 if bs == 32 else (0.05 if bs == 64 else 0.1)
    for ep in epochs_list:
        for lr in learning_rates:
            experiments.append({"BS": bs, "LR": lr, "EP": ep, "DR": dr})

results_csv = "sensitivity_results_updated.csv"

# Load existing results to save time
existing_configs = {}
for source_file in ["sensitivity_results.csv", "sensitivity_results_updated.csv"]:
    if os.path.exists(source_file):
        try:
            df = pd.read_csv(source_file)
            for _, row in df.iterrows():
                try:
                    key = (int(row['Batch size']), float(row['Learning rate']), int(row['Epoch']))
                    existing_configs[key] = {
                        "Batch size": key[0],
                        "Learning rate": key[1],
                        "Epoch": key[2],
                        "dropout": float(row.get('dropout', row.get('Dropout', 0.1))),
                        "R2": float(row.get('R2', row.get('R2 Score', 0))),
                        "RMSE": float(row.get('RMSE', 0)),
                        "MAE": float(row.get('MAE', 0)),
                        "MAE Imp%": row.get('MAE Imp%', '0%')
                    }
                except:
                    continue
        except:
            continue

print(f"Starting execution of 15 EXTRA experiments (Total Target: 36) for Battery {BATTERY}...")
print(f"Adding Missing Epoch 50s and New Epoch 100 set...")
print(f"Already completed: {len([k for k in existing_configs if k in [(e['BS'], e['LR'], e['EP']) for e in experiments]])}/36")
print("-" * 80)

all_results = []

for i, exp in enumerate(experiments):
    bs, lr, ep, dr = exp["BS"], exp["LR"], exp["EP"], exp["DR"]
    key = (bs, lr, ep)
    
    if key in existing_configs:
        # print(f"Exp {i+1}/36: BS={bs}, LR={lr}, Ep={ep} [ALREADY DONE]")
        all_results.append(existing_configs[key])
        continue

    print(f"Exp {i+1}/36: BS={bs}, LR={lr}, Ep={ep}, Dropout={dr} [RUNNING EXTRA...]")
    
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
            
            # Update file incrementally
            pd.DataFrame(all_results).to_csv(results_csv, index=False)
        else:
            print(f"  Error: Results file {res_path} not found.")
    except Exception as e:
        print(f"  Failed: {e}")

# Final Table Generation
final_df = pd.DataFrame(all_results)
if not final_df.empty:
    final_df = final_df.sort_values(by=["Batch size", "Epoch", "Learning rate"])
    final_df.insert(0, 'Case', range(1, len(final_df) + 1))
    
    final_df.to_csv(results_csv, index=False)
    final_df.to_excel(results_csv.replace(".csv", ".xlsx"), index=False)
    
    print("\nSuccess! All 36 variations are now complete.")
    print("-" * 80)
    print(final_df.to_string(index=False))
else:
    print("\nProcess finished with no new data.")
