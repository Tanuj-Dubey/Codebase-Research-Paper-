import os
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# --- BASE PAPER METRICS (Shivendu Mishra et al. - B0005) ---
BASE_MAE = 0.0097
BASE_RMSE = 0.0127

# Hyperparameter grid
experiments = []
for bs in [32, 64, 128]:
    for ep in [20, 30, 50]:
        for lr in [0.0001, 0.001, 0.01]:
            dr = 0.02 if bs == 32 else (0.05 if bs == 64 else 0.1)
            if ep == 50 and lr != 0.0001:
                continue 
            experiments.append({"BS": bs, "LR": lr, "EP": ep, "DR": dr})

battery = "B0005"
results = []

print(f"Starting Final Sensitivity Analysis for Battery {battery}...")
print(f"Comparing against Base Paper (MAE: {BASE_MAE}, RMSE: {BASE_RMSE})")
print("-" * 80)

for i, exp in enumerate(experiments):
    bs, lr, ep, dr = exp["BS"], exp["LR"], exp["EP"], exp["DR"]
    print(f"Exp {i+1}/{len(experiments)}: BS={bs}, LR={lr}, Ep={ep}, Dropout={dr}")
    
    env = os.environ.copy()
    env["BATCH_SIZE"] = str(bs)
    env["LEARNING_RATE"] = str(lr)
    env["EPOCHS"] = str(ep)
    env["DROPOUT"] = str(dr)
    env["CHRONOS_QUIET"] = "1"
    env["HYBRID_MODE"] = "1"
    
    try:
        t0 = time.time()
        # Run silently but check for errors
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
            rmse_imp = ((BASE_RMSE - rmse) / BASE_RMSE) * 100
            
            results.append({
                "Case": i + 1,
                "Batch size": bs,
                "Learning rate": lr,
                "Epoch": ep,
                "Dropout": dr,
                "R2 Score": r2,
                "RMSE": rmse,
                "MAE": mae,
                "MAE Imp%": f"{mae_imp:.2f}%",
                "RMSE Imp%": f"{rmse_imp:.2f}%"
            })
            print(f"  Done in {dt:.1f}s | MAE: {mae:.6f} ({mae_imp:+.2f}%) | RMSE: {rmse:.6f}")
        else:
            print(f"  Error: Results file {res_path} not found.")
    except Exception as e:
        print(f"  Failed: {e}")

# Save final results
results_df = pd.DataFrame(results)
results_df.to_csv("sensitivity_results.csv", index=False)
results_df.to_excel("sensitivity_results.xlsx", index=False)

print("\nFinal Sensitivity Analysis Complete!")
print("-" * 80)
print(results_df.to_string(index=False))
print(f"\nFinal table saved to sensitivity_results.csv and sensitivity_results.xlsx")
