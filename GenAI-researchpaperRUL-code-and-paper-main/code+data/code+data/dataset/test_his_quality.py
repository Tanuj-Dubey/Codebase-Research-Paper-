"""
test_his_quality.py
===================
Validates the statistical quality of each HIS combined file by comparing
the synthetic portion against the real portion.

Benchmarks (must PASS):
    Average Wasserstein Distance  <= 0.0803  (target: "Very Good")
    Average JS Divergence         <= 0.1982  (target: "Good")

Usage:
    python test_his_quality.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, entropy
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────────────────
HIS_DIR   = r"d:\research paper - Copy\code+data\code+data\dataset\HIS"
BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
N_SYNTH   = 6000   # first N_SYNTH rows = synthetic; rest = real

BENCHMARK_WD = 0.0803
BENCHMARK_JS = 0.1982

FEATURE_COLS = ["capacity", "IC_C_H", "IC_C_P", "IC_D_H", "IC_D_P"]


# ── Statistical helpers ───────────────────────────────────────────────────────

def jensen_shannon_divergence(P, Q):
    _P = P / np.linalg.norm(P, ord=1)
    _Q = Q / np.linalg.norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))


def compute_metrics(real_arr, synth_arr, bins=50):
    """Return (wasserstein, js_divergence) for a single feature."""
    scaler = MinMaxScaler()
    real_scaled  = scaler.fit_transform(real_arr.reshape(-1, 1)).flatten()
    synth_scaled = scaler.transform(synth_arr.reshape(-1, 1)).flatten()

    wd = wasserstein_distance(real_scaled, synth_scaled)

    hist_real,  bin_edges = np.histogram(real_scaled,  bins=bins,
                                         range=(0, 1), density=True)
    hist_synth, _         = np.histogram(synth_scaled, bins=bin_edges,
                                         density=True)
    eps = 1e-10
    hist_real  = hist_real  + eps
    hist_synth = hist_synth + eps
    js = jensen_shannon_divergence(hist_real, hist_synth)

    return wd, js


# ── Per-battery test ──────────────────────────────────────────────────────────

def test_battery(battery):
    path = os.path.join(HIS_DIR, "{}_synthetic+Real.csv".format(battery))
    if not os.path.exists(path):
        print("  ERROR: {} not found. Run generate_his_synthetic.py first.".format(path))
        return None, None

    df = pd.read_csv(path)

    synth_df = df.iloc[:N_SYNTH][FEATURE_COLS].dropna()
    real_df  = df.iloc[N_SYNTH:][FEATURE_COLS].dropna()

    print("\n  {}  (synthetic={}, real={})".format(battery, len(synth_df), len(real_df)))
    print("  {:<15} | {:>14} | {:>14}".format("Feature", "Wasserstein", "JS Divergence"))
    print("  " + "-" * 50)

    total_wd = 0.0
    total_js = 0.0

    for col in FEATURE_COLS:
        wd, js = compute_metrics(real_df[col].values, synth_df[col].values)
        total_wd += wd
        total_js += js
        print("  {:<15} | {:>14.4f} | {:>14.4f}".format(col, wd, js))

    avg_wd = total_wd / len(FEATURE_COLS)
    avg_js = total_js / len(FEATURE_COLS)
    print("  " + "-" * 50)
    print("  {:<15} | {:>14.4f} | {:>14.4f}".format("AVERAGE", avg_wd, avg_js))

    return avg_wd, avg_js


# ── Quality label helpers ─────────────────────────────────────────────────────

def wd_label(v):
    if v <= 0.05:    return "Excellent"
    if v <= 0.0803:  return "Very Good"
    if v <= 0.15:    return "Good"
    return "Needs Improvement"


def js_label(v):
    if v <= 0.10:    return "Excellent"
    if v <= 0.1982:  return "Good"
    if v <= 0.35:    return "Moderate"
    return "Poor"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("HIS Synthetic Data Quality Test")
    print("Benchmarks:  Wasserstein <= {}  |  JS Div <= {}".format(
          BENCHMARK_WD, BENCHMARK_JS))
    print("=" * 60)

    all_wd, all_js = [], []

    for battery in BATTERIES:
        avg_wd, avg_js = test_battery(battery)
        if avg_wd is not None:
            all_wd.append(avg_wd)
            all_js.append(avg_js)

    if not all_wd:
        print("\nNo results collected. Aborting.")
        sys.exit(1)

    overall_wd = np.mean(all_wd)
    overall_js = np.mean(all_js)

    print("\n" + "=" * 60)
    print("OVERALL RESULTS (average across all batteries)")
    print("=" * 60)
    print("  Wasserstein:  {:.4f} ({})".format(overall_wd, wd_label(overall_wd)))
    print("  JS Divergence: {:.4f} ({})".format(overall_js, js_label(overall_js)))
    print()

    wd_pass = overall_wd <= BENCHMARK_WD
    js_pass = overall_js <= BENCHMARK_JS

    print("  Wasserstein <= {}: {}".format(BENCHMARK_WD, "PASS" if wd_pass else "FAIL"))
    print("  JS Divergence <= {}: {}".format(BENCHMARK_JS, "PASS" if js_pass else "FAIL"))
    print()

    if wd_pass and js_pass:
        print("  BENCHMARK PASSED -- synthetic data matches quality targets.")
        sys.exit(0)
    else:
        print("  BENCHMARK FAILED -- please re-tune the generator.")
        sys.exit(1)


if __name__ == "__main__":
    main()
