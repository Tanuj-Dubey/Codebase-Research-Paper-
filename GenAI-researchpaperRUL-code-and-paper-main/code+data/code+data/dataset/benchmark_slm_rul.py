"""
benchmark_slm_rul.py
====================
Final benchmark: compares SLM RUL performance on Case-1 (Individual Battery)
against the base paper's Proposed (Informer+HI) and AutoML-22 results.

Runs Case-1 for all 4 batteries, prints a full comparison table,
and asserts we beat AutoML-22 (avg MAE < 0.0336, avg RMSE < 0.0414).

Usage:
    python benchmark_slm_rul.py
"""

import sys
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from train_slm_rul import run_case1, BATTERIES

# ── Ground-truth benchmark numbers from Table 9 of base paper ────────────────
PAPER_RESULTS = {
    # Proposed (Informer + HI)
    "B0005": {"proposed_mae": 0.0090, "proposed_rmse": 0.0129,
              "automl_mae":  0.0283, "automl_rmse":  0.0337},
    "B0006": {"proposed_mae": 0.0304, "proposed_rmse": 0.0330,
              "automl_mae":  0.0221, "automl_rmse":  0.0340},
    "B0007": {"proposed_mae": 0.0159, "proposed_rmse": 0.0184,
              "automl_mae":  0.0361, "automl_rmse":  0.0407},
    "B0018": {"proposed_mae": 0.0188, "proposed_rmse": 0.0212,
              "automl_mae":  0.0479, "automl_rmse":  0.0573},
}

AUTOML_AVG_MAE  = 0.0336
AUTOML_AVG_RMSE = 0.0414


def main():
    print("=" * 80)
    print("SLM RUL Benchmark -- Case-1 (Individual Battery)")
    print("Comparing with: Proposed (Informer+HI) and AutoML-22")
    print("=" * 80)

    slm_results = {}
    for battery in BATTERIES:
        print("\nTraining Case-1 for {}...".format(battery))
        mae, rmse, _ = run_case1(battery, verbose=False)
        slm_results[battery] = (mae, rmse)
        print("  {} -> MAE: {:.4f}  RMSE: {:.4f}".format(battery, mae, rmse))

    # ── Print full comparison table ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE  (Case-1, Individual Battery)")
    print("=" * 80)

    header = "{:<8} | {:>9} {:>9} | {:>13} {:>9} | {:>10} {:>10}".format(
        "Battery",
        "SLM MAE", "SLM RMSE",
        "Proposed MAE", "Prop RMSE",
        "AutoML MAE", "AutoML RMSE"
    )
    print(header)
    print("-" * 80)

    slm_maes, slm_rmses = [], []
    for battery in BATTERIES:
        slm_mae, slm_rmse = slm_results[battery]
        p = PAPER_RESULTS[battery]
        slm_maes.append(slm_mae)
        slm_rmses.append(slm_rmse)

        # Mark SLM vs AutoML-22
        mae_tag  = " <AUTO" if slm_mae  < p["automl_mae"]  else "      "
        rmse_tag = " <AUTO" if slm_rmse < p["automl_rmse"] else "      "

        row = "{:<8} | {:>9.4f} {:>9.4f} | {:>13.4f} {:>9.4f} | {:>10.4f} {:>10.4f}{}{}".format(
            battery,
            slm_mae,    slm_rmse,
            p["proposed_mae"], p["proposed_rmse"],
            p["automl_mae"],   p["automl_rmse"],
            mae_tag, rmse_tag
        )
        print(row)

    avg_slm_mae  = np.mean(slm_maes)
    avg_slm_rmse = np.mean(slm_rmses)
    print("-" * 80)
    avg_row = "{:<8} | {:>9.4f} {:>9.4f} | {:>13.4f} {:>9.4f} | {:>10.4f} {:>10.4f}".format(
        "AVG",
        avg_slm_mae, avg_slm_rmse,
        0.0185, 0.0214,
        AUTOML_AVG_MAE, AUTOML_AVG_RMSE
    )
    print(avg_row)

    # ── PASS / FAIL verdict ──────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    wd_pass  = avg_slm_mae  < AUTOML_AVG_MAE
    rms_pass = avg_slm_rmse < AUTOML_AVG_RMSE

    print("  Avg MAE  {:.4f} < {:.4f} (AutoML-22): {}".format(
          avg_slm_mae,  AUTOML_AVG_MAE,  "PASS" if wd_pass  else "FAIL"))
    print("  Avg RMSE {:.4f} < {:.4f} (AutoML-22): {}".format(
          avg_slm_rmse, AUTOML_AVG_RMSE, "PASS" if rms_pass else "FAIL"))
    print()

    if wd_pass and rms_pass:
        print("  BENCHMARK PASSED -- SLM beats AutoML-22 baseline.")
        sys.exit(0)
    else:
        print("  BENCHMARK FAILED -- SLM does not beat AutoML-22.")
        sys.exit(1)


if __name__ == "__main__":
    main()
