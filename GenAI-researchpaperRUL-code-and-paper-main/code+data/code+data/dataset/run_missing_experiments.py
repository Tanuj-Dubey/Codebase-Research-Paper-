import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# --- BASE PAPER METRICS (Shivendu Mishra et al. - B0005) ---
BASE_MAE = 0.0097
BASE_RMSE = 0.0127

missing_experiments = []
for bs in [64, 128]:
    dr = 0.05 if bs == 64 else 0.1
    for ep in [20, 30, 50]:
        for lr in [0.0001, 0.001, 0.01]:
            if ep == 50 and lr != 0.0001:
                continue
            # skip the ones we already have in the original 9
            if ep == 30 and lr == 0.001:
                continue
            missing_experiments.append({"BS": bs, "LR": lr, "EP": ep, "DR": dr})

battery = "B0005"
results = []

print(f"Starting Execution for BS 64 and 128 Variations...")
print(f"Comparing against Base Paper (MAE: {BASE_MAE})")
print("-" * 80)

for i, exp in enumerate(missing_experiments):
    bs, lr, ep, dr = exp["BS"], exp["LR"], exp["EP"], exp["DR"]
    print(f"Exp {i+1}/{len(missing_experiments)}: BS={bs}, LR={lr}, Ep={ep}, Dropout={dr}")
    
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(bs)
    env["LEARNING_RATE"] = str(lr)
    env["EPOCHS"] = str(ep)
    env["DROPOUT"] = str(dr)
    env["CHRONOS_QUIET"] = "1"
    env["HYBRID_MODE"] = "1"
    
    try:
        t0 = time.time()
        subprocess.run(["python", "hybrid_pipeline.py", battery], env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        dt = time.time() - t0
        
        res_path = f"HIS/final_SOTA_{battery}.csv"
        if os.path.exists(res_path):
            df = pd.read_csv(res_path)
            mae = mean_absolute_error(df['true'], df['pred'])
            rmse = np.sqrt(mean_squared_error(df['true'], df['pred']))
            r2 = r2_score(df['true'], df['pred'])
            
            # Improvement calculation
            mae_imp = ((BASE_MAE - mae) / BASE_MAE) * 100
            
            results.append({
                "Batch size": bs,
                "Learning rate": lr,
                "Epoch": ep,
                "Dropout": dr,
                "R2 Score": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAE Imp%": f"{mae_imp:.2f}%"
            })
            print(f"  Done | MAE: {mae:.6f} ({mae_imp:+.2f}%) | Time: {dt:.1f}s")
        else:
            print(f"  Error: Results file {res_path} not found.")
    except Exception as e:
        print(f"  Failed: {e}")

# Append to original sensitivity_results.csv
if os.path.exists("sensitivity_results.csv"):
    old_df = pd.read_csv("sensitivity_results.csv")
    # Clean up old BS 64/128 rows if we want to replace them with full grid, 
    # but the user asked to "add further", so we just append.
    new_df = pd.DataFrame(results)
    
    if not new_df.empty:
        # Merge and remove duplicates if any (by BS, LR, EP)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        
        # recreate Case column
        combined['Case'] = range(1, len(combined) + 1)
        
        # Save
        combined.to_csv("sensitivity_results.csv", index=False)
        combined.to_excel("sensitivity_results.xlsx", index=False)
        print("\nSuccess! Results updated in sensitivity_results.csv and .xlsx")
        print(combined.to_string(index=False))
else:
    pd.DataFrame(results).to_csv("sensitivity_results.csv", index=False)
    print("\nFile created: sensitivity_results.csv")
