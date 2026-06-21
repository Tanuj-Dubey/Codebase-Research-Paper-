import os
import subprocess
import pandas as pd
import numpy as np
import json
from rich.console import Console
from rich.table import Table

console = Console()
WORKING_DIR = r"c:\Users\AADITYA COM\OneDrive\Desktop\GenAI-researchpaperRUL-code-and-paper-main\code+data\code+data\dataset"
HIS_DIR = os.path.join(WORKING_DIR, "HIS")

# --- Define Sensitivity Grid ---
SWEEP_CONFIG = {
    "ALPHA": [0.0, 0.5, 0.85, 1.0],         # Gating strength
    "LAMBDA": [0.0, 1e-4, 1e-3, 1e-2],    # Ridge regularization
    "CONTEXTS": ["96", "96,48", "96,64,48"] # Chronos context spans
}

RESULTS = []

def run_experiment(case_num, alpha, lambd, contexts, battery="B0005"):
    env = os.environ.copy()
    env["CASE3_RESIDUAL_GATE_ALPHA"] = str(alpha)
    env["AFFINE_RIDGE_LAMBDA"] = str(lambd)
    env["CHRONOS_CONTEXTS"] = contexts
    env["CHRONOS_NUM_SAMPLES"] = "8" # Fast sweep for demo
    env["CASE3_RESIDUAL_EPOCHS"] = "2" # Smoke test for sweep
    env["CHRONOS_QUIET"] = "1"
    
    script_map = {
        1: "hybrid_pipeline.py",
        2: "hybrid_pipeline_case2.py",
        3: "hybrid_model_case3.py"
    }
    
    script = script_map[case_num]
    # For Case 1 we pass battery ID, for others we pass fold index 0
    arg = "0" if case_num > 1 else battery
    cmd = ["python", script, arg]
    
    console.print(f"[yellow]Running Case {case_num}: α={alpha}, λ={lambd}, L=[{contexts}][/yellow]")
    
    try:
        # We run silently but capture output to verify success
        subprocess.run(cmd, cwd=WORKING_DIR, env=env, check=True, capture_output=True, text=True)
        
        # After run finishes, we read the results from the HIS directory
        if case_num == 3:
            res_df = pd.read_csv(os.path.join(HIS_DIR, "case3_results.csv"))
            last_row = res_df.iloc[-1]
            return {"mae": last_row["MAE"], "rmse": last_row["RMSE"]}
        elif case_num == 1:
            res_df = pd.read_csv(os.path.join(HIS_DIR, f"final_SOTA_{battery}.csv"))
            # Calculate metrics
            mae = np.mean(np.abs(res_df["true"] - res_df["pred_meta_calibrated"]))
            rmse = np.sqrt(np.mean((res_df["true"] - res_df["pred_meta_calibrated"])**2))
            return {"mae": mae, "rmse": rmse}
    except Exception as e:
        console.print(f"[red]Failed experiment {case_num}: {e}[/red]")
        return {"mae": 0.0, "rmse": 0.0}

def main():
    # Performance Sensitivity Sweep for ALPHA (Case 3)
    console.print("[bold blue]Starting Sensitivity Sweep: Case 3 ALPHA impact[/bold blue]")
    for a in SWEEP_CONFIG["ALPHA"]:
        res = run_experiment(3, a, 1e-3, "96,64,48")
        RESULTS.append({
            "Experiment": f"Case3_Alpha_{a}",
            "Case": 3,
            "Alpha": a,
            "Lambda": 0.001,
            "Contexts": "96,64,48",
            "MAE": res["mae"],
            "RMSE": res["rmse"]
        })
    
    # Context Sensitivity Sweep (Case 1)
    console.print("[bold blue]Starting Sensitivity Sweep: Case 1 Context impact[/bold blue]")
    for ctx in SWEEP_CONFIG["CONTEXTS"]:
        res = run_experiment(1, 0.85, 1e-3, ctx, battery="B0005")
        RESULTS.append({
            "Experiment": f"Case1_Ctx_{ctx}",
            "Case": 1,
            "Alpha": 0.85,
            "Lambda": 0.001,
            "Contexts": ctx,
            "MAE": res["mae"],
            "RMSE": res["rmse"]
        })

    df = pd.DataFrame(RESULTS)
    df.to_csv("hyperparameter_sensitivity_results.csv", index=False)
    console.print("[bold green]Sweep Complete. Results saved to hyperparameter_sensitivity_results.csv[/bold green]")
    
    # Display table
    table = Table(title="IEEE Hyperparameter Sensitivity Analysis")
    table.add_column("Experiment", style="cyan")
    table.add_column("Case", style="white")
    table.add_column("Alpha", style="magenta")
    table.add_column("Contexts", style="yellow")
    table.add_column("MAE (Ah)", style="green")
    
    for r in RESULTS:
        table.add_row(r["Experiment"], str(r["Case"]), str(r["Alpha"]), r["Contexts"], f"{r['MAE']:.6f}")
    
    console.print(table)

if __name__ == "__main__":
    main()
