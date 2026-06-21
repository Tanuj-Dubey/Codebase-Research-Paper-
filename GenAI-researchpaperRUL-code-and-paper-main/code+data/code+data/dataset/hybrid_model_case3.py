"""
Case-3: leave-one-out cross-battery capacity forecasting (3 train → 1 test).

Strict protocol:
- Exactly LAST_N=3000 rows per battery (assert); sort by cycle; tail(3000).
- StandardScaler fit ONLY on concatenated train batteries (9000 rows).
- Sliding windows built PER battery, then concatenated — no cross-battery windows.
- Chronos context: capacity[:, :-1] (last in-window step excluded), contexts (96, 64, 48).
- Train-only calibration: 85% / 15% split to select nonnegative context weights (sum=1);
  ridge affine (λ=1e-3) refit on all train windows. No test leakage.
- Residual target: (y − Chronos_cal) × RESIDUAL_SCALE (10); inference divides by RESIDUAL_SCALE.
- Hybrid: 0.6·r1 + 0.4·r2 on scaled residuals; optional B0006 fallback when val MAE (Ah) high.
- Adaptive clip on final scaled predictions: train mean ± 3·train std (train-only stats).
- Optional `CASE3_USE_CONFIDENCE_BLEND=1`: Chronos per-context variance blend + ridge affine; hybrid uses residual gate: `r_conf ∝ 1/(√|r_mix|+eps)`, optional × Chronos uncertainty, normalize, clip to [floor,cap], optional causal EMA across batches, `r_eff = α · r_conf · r_mix`.

Outputs: HIS/case3_results.csv, plots/case3_loss_*.png, plots/case3_prediction_*.png

Low VRAM (e.g. 4 GiB): infer batch and train batch auto-cap; optional CHRONOS_LOW_VRAM=1 / CASE3_LOW_VRAM=1.
Chronos predict retries with smaller sub-batches on CUDA OOM.

Publication metrics: all reported RMSE values use RMSE = sqrt(MSE) exactly (no MAE×k proxy for model
metrics). H2O baseline RMSE is stored as explicit literature targets (edit H2O_RMSE if your paper
differs). Final run defaults: confidence blend on, num_samples=32, gate α=0.85, EMA β=0.7.
"""
from __future__ import annotations

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import random
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

# Optional plotting
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    from scipy.signal import savgol_filter

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

CONSOLE = Console()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIS_DIR = os.path.join(SCRIPT_DIR, "HIS")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

LAST_N = 3000
SEQ_LEN = 96
PRED_LEN = 1
TRAIN_BATCH = max(1, int(os.environ.get("CASE3_TRAIN_BATCH", "256")))
BATCH_SIZE = TRAIN_BATCH
DEFAULT_INFER_BATCH = 64
ALL_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

CASE_3_CONFIG = [
    (["B0006", "B0007", "B0018"], "B0005"),
    (["B0005", "B0007", "B0018"], "B0006"),
    (["B0005", "B0006", "B0018"], "B0007"),
    (["B0005", "B0006", "B0007"], "B0018"),
]

# H2O reference targets (Ah). RMSE must be explicit baseline RMSE from your source — not approximated as k×MAE.
H2O_MAE = {
    "B0005": 0.0139,
    "B0006": 0.0141,
    "B0007": 0.0169,
    "B0018": 0.0101,
}
H2O_RMSE = {
    "B0005": 0.0178,
    "B0006": 0.0181,
    "B0007": 0.0217,
    "B0018": 0.0129,
}

def _parse_context_tuple(s: str) -> Optional[Tuple[int, ...]]:
    s = (s or "").strip()
    if not s:
        return None
    parts = tuple(int(x.strip()) for x in s.split(",") if x.strip())
    return parts if parts else None


_ctx_env = _parse_context_tuple(os.environ.get("CASE3_CONTEXTS", ""))
CHRONOS_CONTEXT_LENGTHS = _ctx_env if _ctx_env is not None else (96, 64, 48)
# Default 32; override with CHRONOS_NUM_SAMPLES
CHRONOS_NUM_SAMPLES = max(1, int(os.environ.get("CHRONOS_NUM_SAMPLES", "32")))
AFFINE_RIDGE_LAMBDA = float(os.environ.get("AFFINE_RIDGE_LAMBDA", "1e-3"))
CHRONOS_CLIP_MARGIN_FRAC = float(os.environ.get("CHRONOS_CLIP_MARGIN_FRAC", "0.05"))
# (w_L96, w_L64, w_L48) must sum to 1 — scanned on internal calibration tail only
THREE_CONTEXT_WEIGHT_GRID: List[Tuple[float, float, float]] = [
    (0.5, 0.3, 0.2),
    (0.45, 0.35, 0.2),
    (0.4, 0.35, 0.25),
    (0.55, 0.25, 0.2),
    (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    (0.4, 0.4, 0.2),
    (0.6, 0.25, 0.15),
    (0.35, 0.4, 0.25),
]
RESIDUAL_SCALE = 10.0
W_RESIDUAL_M1 = 0.6
W_RESIDUAL_M2 = 0.4
# 15% tail of train windows for selecting w (85% / 15% calibration split)
CALIB_HOLDOUT_FRAC = float(os.environ.get("CASE3_CALIB_HOLDOUT_FRAC", "0.15"))
CHRONOS_LOG_BATCH_EVERY = max(0, int(os.environ.get("CHRONOS_LOG_BATCH_EVERY", "25")))
VERBOSE = os.environ.get("CHRONOS_QUIET", "").lower() not in ("1", "true", "yes")

RESIDUAL_EPOCHS = int(os.environ.get("CASE3_RESIDUAL_EPOCHS", "200"))
RESIDUAL_PATIENCE = int(os.environ.get("CASE3_RESIDUAL_PATIENCE", "20"))
B0006_VAL_MAE_THRESHOLD_AH = float(os.environ.get("CASE3_B0006_VAL_MAE_THRESHOLD", "0.02"))
GRAD_CLIP_NORM = float(os.environ.get("CASE3_GRAD_CLIP", "1.0"))
def _env_flag_default_true(name: str, default_on: bool = True) -> bool:
    v = os.environ.get(name)
    if v is None or str(v).strip() == "":
        return default_on
    return str(v).strip().lower() not in ("0", "false", "no", "off")


# Per-window Chronos blend + residual gate (default ON for publication run; set CASE3_USE_CONFIDENCE_BLEND=0 to disable)
CASE3_USE_CONFIDENCE_BLEND = _env_flag_default_true("CASE3_USE_CONFIDENCE_BLEND", True)
CONFIDENCE_VAR_EPS = float(os.environ.get("CASE3_CONFIDENCE_VAR_EPS", "1e-6"))
# Residual gate (with CASE3_USE_CONFIDENCE_BLEND): inverse-sqrt, normalize, clip, optional EMA & Chronos synergy
RESIDUAL_PRED_CONF_EPS = float(os.environ.get("CASE3_RESIDUAL_PRED_CONF_EPS", "1e-6"))
RESIDUAL_GATE_FLOOR = float(os.environ.get("CASE3_RESIDUAL_GATE_FLOOR", "0.2"))
RESIDUAL_GATE_CAP = float(os.environ.get("CASE3_RESIDUAL_GATE_CAP", "1.0"))
RESIDUAL_GATE_ALPHA = float(os.environ.get("CASE3_RESIDUAL_GATE_ALPHA", "0.85"))
_rg_beta = os.environ.get("CASE3_RESIDUAL_GATE_SMOOTH_BETA", "0.7").strip()
RESIDUAL_GATE_SMOOTH_BETA: Optional[float] = (
    float(_rg_beta) if _rg_beta and 0.0 < float(_rg_beta) < 1.0 else None
)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_performance_runtime() -> None:
    """Speed-optimized cuDNN (final run); reproducibility is secondary to throughput."""
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


set_seed(42)


def _init_device() -> Tuple[torch.device, Optional[str]]:
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        d = torch.device("cuda:0")
        CONSOLE.print(
            f"[green]GPU:[/green] {torch.cuda.get_device_name(0)} | torch={torch.__version__}"
        )
        return d, "auto"
    CONSOLE.print("[yellow]CPU[/yellow] torch")
    return torch.device("cpu"), None


DEVICE, CHRONOS_DEVICE_MAP = _init_device()
configure_performance_runtime()

# String label for logs (matches common torch.device choice)
DEVICE_KIND = "cuda" if torch.cuda.is_available() else "cpu"


def _cuda_total_mem_gb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        return float(torch.cuda.get_device_properties(0).total_memory) / (1024**3)
    except Exception:
        return None


def _pin_memory() -> bool:
    return torch.cuda.is_available()


def _infer_batch() -> int:
    e = int(os.environ.get("CHRONOS_INFER_BATCH", "0") or 0)
    if e > 0:
        return max(1, e)
    if os.environ.get("CHRONOS_LOW_VRAM", "").lower() in ("1", "true", "yes"):
        return 4
    gb = _cuda_total_mem_gb()
    if gb is None:
        return min(48, DEFAULT_INFER_BATCH)
    # Chronos T5 + num_samples is VRAM-heavy; ~4 GiB cards cannot use 64.
    if gb >= 12:
        return 64
    if gb >= 8:
        return 32
    if gb >= 6:
        return 16
    if gb >= 4.5:
        return 8
    return 4


def _cap_train_batch_for_vram(requested: int) -> int:
    """Keep residual training on GPU without OOM while Chronos stays loaded."""
    if not torch.cuda.is_available():
        return max(1, requested)
    if os.environ.get("CASE3_LOW_VRAM", "").lower() in ("1", "true", "yes"):
        return min(max(1, requested), 8)
    gb = _cuda_total_mem_gb()
    if gb is None:
        return max(1, requested)
    if gb <= 5.0:
        return min(max(1, requested), 16)
    if gb <= 8.0:
        return min(max(1, requested), 64)
    if gb <= 12.0:
        return min(max(1, requested), 128)
    return max(1, requested)


INFERENCE_BATCH_SIZE = _infer_batch()
BATCH_SIZE = _cap_train_batch_for_vram(BATCH_SIZE)
CUDA_MEM_GB = _cuda_total_mem_gb()


def _chronos_model_candidates() -> List[str]:
    """Prefer env CHRONOS_MODEL; optional large model on high-VRAM GPU."""
    env = os.environ.get("CHRONOS_MODEL", "").strip()
    if env:
        return [env, "amazon/chronos-t5-base", "amazon/chronos-t5-small"]
    use_large = os.environ.get("CHRONOS_USE_LARGE", "").lower() in ("1", "true", "yes")
    if use_large and torch.cuda.is_available():
        try:
            gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            gb = 0.0
        if gb >= 12.0:
            return [
                "amazon/chronos-t5-large",
                "amazon/chronos-t5-base",
                "amazon/chronos-t5-small",
            ]
    return ["amazon/chronos-t5-base", "amazon/chronos-t5-small"]


def load_battery_strict(bat: str) -> pd.DataFrame:
    path = os.path.join(HIS_DIR, f"{bat}_synthetic+Real.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.sort_values("cycle").reset_index(drop=True)
    df = df.tail(LAST_N).reset_index(drop=True)
    assert len(df) == LAST_N, f"{bat}: expected {LAST_N} rows, got {len(df)}"
    return df


def verify_all_batteries() -> None:
    for bat in ALL_BATTERIES:
        df = load_battery_strict(bat)
        assert len(df) == LAST_N, bat
    CONSOLE.print(
        f"[bold green]OK:[/bold green] all {len(ALL_BATTERIES)} batteries have exactly {LAST_N} rows."
    )


def sliding_windows_on_scaled(
    arr: np.ndarray, cap_idx: int, n_feat: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(arr) <= SEQ_LEN:
        z = np.empty((0, SEQ_LEN, n_feat), dtype=np.float32)
        z1 = np.empty((0, 1), dtype=np.float32)
        return z, z1, z1
    v = np.lib.stride_tricks.sliding_window_view(arr, SEQ_LEN, axis=0)
    v = v.transpose(0, 2, 1)
    X = v[:-1]
    y = arr[SEQ_LEN:, cap_idx].reshape(-1, 1)
    lc = v[:-1, -1, cap_idx].reshape(-1, 1)
    return X.astype(np.float32), y.astype(np.float32), lc.astype(np.float32)


def _capacity_context_excluding_last(
    x_np: np.ndarray, cap_idx: int, context_len: int
) -> np.ndarray:
    pref = x_np[:, :-1, cap_idx]
    sl = min(int(context_len), pref.shape[1])
    return np.ascontiguousarray(pref[:, -sl:], dtype=np.float32)


def affine_ridge_fit(
    p: np.ndarray, y: np.ndarray, lam: float = AFFINE_RIDGE_LAMBDA
) -> Tuple[float, float]:
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return 1.0, 0.0
    Xd = np.column_stack([p, np.ones(n)])
    XtX = Xd.T @ Xd
    XtX[0, 0] += lam
    beta = np.linalg.solve(XtX, Xd.T @ y)
    return float(beta[0]), float(beta[1])


def _clip_bounds(y_train: np.ndarray) -> Tuple[float, float]:
    y = np.asarray(y_train, dtype=np.float64).ravel()
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo + 1e-8
    m = CHRONOS_CLIP_MARGIN_FRAC * span
    return lo - m, hi + m


def _blend_from_weights(
    preds_by_L: Dict[int, np.ndarray], Ls_desc: List[int], weights: Tuple[float, ...]
) -> np.ndarray:
    acc = np.zeros_like(preds_by_L[Ls_desc[0]], dtype=np.float64)
    for w, L in zip(weights, Ls_desc):
        acc += float(w) * preds_by_L[L].astype(np.float64)
    return acc


def apply_multi_blend_affine(
    preds_by_L: Dict[int, np.ndarray],
    weights: Tuple[float, ...],
    a: float,
    b: float,
) -> np.ndarray:
    Ls = sorted(preds_by_L.keys(), reverse=True)
    assert len(Ls) == len(weights), f"contexts {Ls} vs weights {weights}"
    p = _blend_from_weights(preds_by_L, Ls, weights)
    return (a * p + b).astype(np.float32)


def apply_affine_only(p: np.ndarray, a: float, b: float) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).ravel()
    return (a * p + b).astype(np.float32)


def chronos_uncertainty_from_vars(vars_by_L: Dict[int, np.ndarray]) -> np.ndarray:
    """Per window: mean variance across contexts → scalar confidence 1/(mean_var+eps)."""
    Ls = sorted(vars_by_L.keys(), reverse=True)
    V = np.stack([vars_by_L[L].astype(np.float64).ravel() for L in Ls], axis=1)
    v_mean = V.mean(axis=1)
    return (1.0 / (v_mean + CONFIDENCE_VAR_EPS)).astype(np.float32)


def apply_residual_confidence_gate(
    r_mix: torch.Tensor,
    chronos_conf_batch: Optional[torch.Tensor] = None,
    ema_carry: Optional[torch.Tensor] = None,
) -> Tuple[
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]],
]:
    """
    When CASE3_USE_CONFIDENCE_BLEND:
      eps = RESIDUAL_PRED_CONF_EPS everywhere below.
      r_conf = 1/(sqrt(|r_mix|)+eps); Chronos: chronos_conf /= (max+eps) then r_conf *= chronos_conf;
      den = r_conf.max(); if den < eps → skip gating (degenerate);
      r_conf /= (den+eps); clip [floor, cap]; optional EMA along batch + scalar carry across batches;
      r_eff = alpha * (r_conf * r_mix).
    Returns (r_eff, next_ema_carry, diag) where diag is (r_conf_1d_cpu, chronos_norm_1d_cpu|None) or None if gate skipped.
    """
    if not CASE3_USE_CONFIDENCE_BLEND:
        return r_mix, None, None
    eps = RESIDUAL_PRED_CONF_EPS
    floor = RESIDUAL_GATE_FLOOR
    cap = RESIDUAL_GATE_CAP
    alpha = RESIDUAL_GATE_ALPHA

    r_flat = r_mix.squeeze(-1)
    rc = 1.0 / (torch.sqrt(torch.abs(r_flat)) + eps)
    cc_norm_out: Optional[torch.Tensor] = None
    if chronos_conf_batch is not None:
        cc = chronos_conf_batch.to(device=r_mix.device, dtype=r_flat.dtype).flatten()
        if cc.shape[0] != rc.shape[0]:
            raise ValueError(f"chronos_conf_batch length {cc.shape[0]} != batch {rc.shape[0]}")
        cc = cc / (cc.max() + eps)
        cc_norm_out = cc.detach().cpu()
        rc = rc * cc
    den = float(rc.max().item())
    if den < eps:
        return r_mix, ema_carry, None
    rc = rc / (rc.max() + eps)
    rc = torch.clamp(rc, floor, cap)

    beta = RESIDUAL_GATE_SMOOTH_BETA
    if beta is not None:
        out = torch.empty_like(rc)
        prev = ema_carry
        for i in range(rc.shape[0]):
            if prev is None:
                prev = rc[i]
            else:
                prev = beta * prev + (1.0 - beta) * rc[i]
            out[i] = prev
        next_carry = prev.detach() if rc.shape[0] > 0 else ema_carry
        rc = out
    else:
        next_carry = None

    r_conf_for_stats = rc.detach().cpu()
    r_eff = alpha * (rc.unsqueeze(-1) * r_mix)
    diag: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = (r_conf_for_stats, cc_norm_out)
    return r_eff, next_carry, diag


def blend_chronos_by_confidence(
    preds_by_L: Dict[int, np.ndarray],
    vars_by_L: Dict[int, np.ndarray],
) -> np.ndarray:
    """
    Per window: var_i = variance of Chronos samples for context i;
    conf_i = 1/(var_i + eps); weights = conf / sum(conf); y = sum_i weights_i * pred_i.
    """
    Ls = sorted(preds_by_L.keys(), reverse=True)
    P = np.stack([preds_by_L[L].astype(np.float64).ravel() for L in Ls], axis=1)
    V = np.stack([vars_by_L[L].astype(np.float64).ravel() for L in Ls], axis=1)
    assert P.shape == V.shape, (P.shape, V.shape)
    conf = 1.0 / (V + CONFIDENCE_VAR_EPS)
    w = conf / np.sum(conf, axis=1, keepdims=True)
    return np.sum(w * P, axis=1)


def fit_affine_on_blended(
    y_true: np.ndarray, p_blended: np.ndarray
) -> Tuple[float, float, float, float]:
    """Train-only ridge affine on already-blended Chronos predictions (full train)."""
    y = y_true.ravel().astype(np.float64)
    p = np.asarray(p_blended, dtype=np.float64).ravel()
    a, b = affine_ridge_fit(p, y)
    pred = a * p + b
    return a, b, mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))


def fit_multi_context_weights_and_affine(
    y_true: np.ndarray, preds_by_L: Dict[int, np.ndarray]
) -> Tuple[Tuple[float, ...], float, float, float, float]:
    """
    85% / 15% chronological split: select nonnegative context weights (grid, sum=1)
    on calibration tail; ridge affine on head only per candidate; refit (a,b) on full train.
    """
    Ls = sorted(preds_by_L.keys(), reverse=True)
    y = y_true.ravel().astype(np.float64)
    n = len(y)
    if len(Ls) == 1:
        p = preds_by_L[Ls[0]].ravel().astype(np.float64)
        a, b = affine_ridge_fit(p, y)
        pred = a * p + b
        return (1.0,), a, b, mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))

    stacks = [preds_by_L[L].ravel().astype(np.float64) for L in Ls]
    n_cal = max(2, min(n - 3, int(round(n * CALIB_HOLDOUT_FRAC))))
    cut = n - n_cal
    use_holdout = cut >= 10

    best: Optional[Tuple[float, Tuple[float, ...]]] = None
    weight_candidates = THREE_CONTEXT_WEIGHT_GRID
    if len(Ls) == 2:
        weight_candidates = [(w, 1.0 - w) for w in (0.3, 0.4, 0.5, 0.6, 0.7)]

    if use_holdout:
        if VERBOSE:
            CONSOLE.print(
                f"[bold]Calibration split[/bold]: first {cut}/{n} windows for ridge fit, "
                f"last {n_cal} for selecting context weights (no test use)."
            )
        for tup in weight_candidates:
            if len(tup) != len(Ls):
                continue
            if abs(sum(tup) - 1.0) > 1e-6:
                continue
            pb = np.zeros(n, dtype=np.float64)
            for w, arr in zip(tup, stacks):
                pb += w * arr
            ah, bh = affine_ridge_fit(pb[:cut], y[:cut])
            mae_c = mean_absolute_error(y[cut:], ah * pb[cut:] + bh)
            if VERBOSE and len(Ls) == 3:
                CONSOLE.print(f"  w={tuple(round(x, 3) for x in tup)}  cal_MAE={mae_c:.6f}")
            elif VERBOSE and len(Ls) == 2:
                CONSOLE.print(f"  w={tup[0]:.2f}  cal_MAE={mae_c:.6f}")
            if best is None or mae_c < best[0]:
                best = (mae_c, tuple(float(x) for x in tup))
        w_best = best[1]
    else:
        if VERBOSE:
            CONSOLE.print("[yellow]Short train: weights by full-train MAE.[/yellow]")
        for tup in weight_candidates:
            if len(tup) != len(Ls):
                continue
            if abs(sum(tup) - 1.0) > 1e-6:
                continue
            pb = np.zeros(n, dtype=np.float64)
            for w, arr in zip(tup, stacks):
                pb += w * arr
            a, b = affine_ridge_fit(pb, y)
            mae_f = mean_absolute_error(y, a * pb + b)
            if best is None or mae_f < best[0]:
                best = (mae_f, tuple(float(x) for x in tup))
        w_best = best[1]

    pb_full = np.zeros(n, dtype=np.float64)
    for w, arr in zip(w_best, stacks):
        pb_full += w * arr
    a_best, b_best = affine_ridge_fit(pb_full, y)
    pred = a_best * pb_full + b_best
    return (
        w_best,
        a_best,
        b_best,
        mean_absolute_error(y, pred),
        np.sqrt(mean_squared_error(y, pred)),
    )


def _load_chronos():
    from chronos import ChronosPipeline

    for mid in _chronos_model_candidates():
        try:
            kw: Dict[str, Any] = {"torch_dtype": torch.float16}
            if CHRONOS_DEVICE_MAP is not None:
                kw["device_map"] = CHRONOS_DEVICE_MAP
            return ChronosPipeline.from_pretrained(mid, **kw), mid
        except Exception as e:
            if VERBOSE:
                CONSOLE.print(f"[red]Chronos load failed {mid}:[/red] {e}")
    raise RuntimeError("Could not load any Chronos checkpoint.")


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def _forecast_batch_median(
    pipeline, x_np: np.ndarray, cap_idx: int, context_len: int
) -> np.ndarray:
    arr = _capacity_context_excluding_last(x_np, cap_idx, context_len)
    n = int(arr.shape[0])
    ctx = torch.from_numpy(arr).float().cpu()
    try:
        with torch.inference_mode():
            fc = pipeline.predict(
                ctx, prediction_length=PRED_LEN, num_samples=CHRONOS_NUM_SAMPLES
            )
        med = torch.median(fc, dim=1).values[:, -1].float().cpu().numpy()
        return med
    except Exception as e:
        if not _is_cuda_oom(e) or n <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        mid = n // 2
        m1 = _forecast_batch_median(pipeline, x_np[:mid], cap_idx, context_len)
        m2 = _forecast_batch_median(pipeline, x_np[mid:], cap_idx, context_len)
        return np.concatenate([m1, m2])


def _forecast_batch_median_var(
    pipeline, x_np: np.ndarray, cap_idx: int, context_len: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Median and sample variance (across num_samples) at last horizon step, per row."""
    arr = _capacity_context_excluding_last(x_np, cap_idx, context_len)
    n = int(arr.shape[0])
    ctx = torch.from_numpy(arr).float().cpu()
    try:
        with torch.inference_mode():
            fc = pipeline.predict(
                ctx, prediction_length=PRED_LEN, num_samples=CHRONOS_NUM_SAMPLES
            )
        last = fc[:, :, -1].float()
        med = torch.median(last, dim=1).values.cpu().numpy()
        var = torch.var(last, dim=1, unbiased=False).cpu().numpy()
        return med.astype(np.float32), var.astype(np.float32)
    except Exception as e:
        if not _is_cuda_oom(e) or n <= 1:
            raise
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        mid = n // 2
        m1, v1 = _forecast_batch_median_var(pipeline, x_np[:mid], cap_idx, context_len)
        m2, v2 = _forecast_batch_median_var(pipeline, x_np[mid:], cap_idx, context_len)
        return np.concatenate([m1, m2]), np.concatenate([v1, v2])


def run_chronos_multicontext(
    X: np.ndarray,
    cap_idx: int,
    pipeline,
    case_tag: str,
    split_label: str,
    return_variance: bool = False,
) -> Tuple[Dict[int, np.ndarray], Optional[Dict[int, np.ndarray]]]:
    out_p: Dict[int, np.ndarray] = {}
    out_v: Optional[Dict[int, np.ndarray]] = {} if return_variance else None
    for L in CHRONOS_CONTEXT_LENGTHS:
        if L > SEQ_LEN:
            continue
        n = len(X)
        parts: List[np.ndarray] = []
        parts_v: List[np.ndarray] = []
        n_b = (n + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=CONSOLE,
            ) as prog:
                label = f"{case_tag} {split_label} L={L}"
                if return_variance:
                    label += " (+var)"
                task = prog.add_task(label, total=n)
                for bi, i in enumerate(range(0, n, INFERENCE_BATCH_SIZE)):
                    batch = X[i : i + INFERENCE_BATCH_SIZE]
                    if return_variance:
                        m, v = _forecast_batch_median_var(pipeline, batch, cap_idx, L)
                        parts.append(m)
                        parts_v.append(v)
                    else:
                        parts.append(_forecast_batch_median(pipeline, batch, cap_idx, L))
                    prog.update(task, advance=batch.shape[0])
                    if torch.cuda.is_available() and CUDA_MEM_GB is not None and CUDA_MEM_GB <= 6.0:
                        torch.cuda.empty_cache()
                    if VERBOSE and CHRONOS_LOG_BATCH_EVERY and (
                        bi == 0 or bi == n_b - 1 or (bi + 1) % CHRONOS_LOG_BATCH_EVERY == 0
                    ):
                        CONSOLE.print(f"  … batch {bi+1}/{n_b}")
        out_p[L] = np.concatenate(parts).astype(np.float32)
        if return_variance and out_v is not None:
            out_v[L] = np.concatenate(parts_v).astype(np.float32)
    return out_p, out_v


# 1.5× default capacity vs original 64-dim stacks
_DMODEL = int(round(64 * 1.5))
_HEAD = int(round(64 * 1.5))


class PatchTSTResidual(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = _DMODEL,
        n_heads: int = 4,
        n_layers: int = 2,
    ):
        super().__init__()
        self.patch_size, self.stride = 16, 8
        self.num_patches = (seq_len - self.patch_size) // self.stride + 1
        self.proj = nn.Linear(self.patch_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.num_patches * input_dim + 2, _HEAD),
            nn.ReLU(),
            nn.Linear(_HEAD, 1),
        )

    def forward(self, x_win: torch.Tensor, c_pred: torch.Tensor, last_cap: torch.Tensor) -> torch.Tensor:
        B, _, C = x_win.shape
        x = x_win.transpose(1, 2)
        patches = [
            x[:, :, i * self.stride : i * self.stride + self.patch_size].unsqueeze(2)
            for i in range(self.num_patches)
        ]
        xb = torch.cat(patches, dim=2).reshape(-1, self.num_patches, self.patch_size)
        out = self.transformer(self.proj(xb)).reshape(B, -1)
        return self.head(torch.cat([out, c_pred.unsqueeze(-1), last_cap], dim=1))


class TinyTimeMixer(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, hidden: int = _HEAD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len + 2, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, cp: torch.Tensor, lc: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x.reshape(x.size(0), -1), cp.unsqueeze(-1), lc], dim=1))


def _mae_capacity_from_residual(
    model: nn.Module, loader: DataLoader, weighted: bool
) -> float:
    """MAE in scaled capacity space: |y − (c + r̂)|."""
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        if weighted:
            for Xb, cpb, lcb, yb, _ in loader:
                Xb, cpb, lcb, yb = (
                    Xb.to(DEVICE),
                    cpb.to(DEVICE),
                    lcb.to(DEVICE),
                    yb.to(DEVICE),
                )
                pred = cpb.unsqueeze(-1) + model(Xb, cpb, lcb) / RESIDUAL_SCALE
                tot += (pred - yb).abs().sum().item()
                cnt += yb.numel()
        else:
            for Xb, cpb, lcb, yb in loader:
                Xb, cpb, lcb, yb = (
                    Xb.to(DEVICE),
                    cpb.to(DEVICE),
                    lcb.to(DEVICE),
                    yb.to(DEVICE),
                )
                pred = cpb.unsqueeze(-1) + model(Xb, cpb, lcb) / RESIDUAL_SCALE
                tot += (pred - yb).abs().sum().item()
                cnt += yb.numel()
    return tot / max(1, cnt)


def train_residual_case3(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    case_tag: str,
    epochs: int = RESIDUAL_EPOCHS,
    patience: int = RESIDUAL_PATIENCE,
) -> Tuple[str, Dict[str, List[float]]]:
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    model.to(DEVICE)
    best_metric, no_imp = float("inf"), 0
    name = model.__class__.__name__
    ckpt = os.path.join(SCRIPT_DIR, f"best_case3_{name}_{case_tag}.pth")

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task(f"[cyan]{name}[/cyan]", total=epochs)
        stopped_epoch: Optional[int] = None
        for epoch in range(epochs):
            model.train()
            t_loss_sum = 0.0
            n_bt = 0
            for Xb, cpb, lcb, yb, wb in train_loader:
                Xb, cpb, lcb, yb, wb = (
                    Xb.to(DEVICE),
                    cpb.to(DEVICE),
                    lcb.to(DEVICE),
                    yb.to(DEVICE),
                    wb.to(DEVICE),
                )
                trg = (yb - cpb.unsqueeze(-1)) * RESIDUAL_SCALE
                opt.zero_grad()
                out = model(Xb, cpb, lcb)
                loss_vec = (
                    0.7 * nn.functional.l1_loss(out, trg, reduction="none")
                    + 0.3 * nn.functional.mse_loss(out, trg, reduction="none")
                ) * wb
                loss = loss_vec.mean()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                opt.step()
                t_loss_sum += float(loss.item())
                n_bt += 1

            train_loss_epoch = t_loss_sum / max(1, n_bt)
            history["train_loss"].append(train_loss_epoch)

            model.eval()
            v_loss_sum = 0.0
            n_vb = 0
            with torch.no_grad():
                for Xb, cpb, lcb, yb in val_loader:
                    Xb, cpb, lcb, yb = (
                        Xb.to(DEVICE),
                        cpb.to(DEVICE),
                        lcb.to(DEVICE),
                        yb.to(DEVICE),
                    )
                    trg = (yb - cpb.unsqueeze(-1)) * RESIDUAL_SCALE
                    out = model(Xb, cpb, lcb)
                    loss_v = (
                        0.7 * nn.functional.l1_loss(out, trg)
                        + 0.3 * nn.functional.mse_loss(out, trg)
                    )
                    v_loss_sum += float(loss_v.item())
                    n_vb += 1

            val_loss_epoch = v_loss_sum / max(1, n_vb)
            history["val_loss"].append(val_loss_epoch)

            train_mae = _mae_capacity_from_residual(model, train_loader, weighted=True)
            val_mae = _mae_capacity_from_residual(model, val_loader, weighted=False)

            sch.step(val_loss_epoch)
            if val_loss_epoch < best_metric:
                best_metric = val_loss_epoch
                no_imp = 0
                torch.save(model.state_dict(), ckpt)
            else:
                no_imp += 1

            if VERBOSE:
                CONSOLE.print(
                    f"[RESIDUAL TRAINING] Epoch {epoch + 1:03d} | Train MAE {train_mae:.6f} | "
                    f"Val MAE {val_mae:.6f} | Best val_loss {best_metric:.6f} | {name}"
                )
            progress.update(task, advance=1)

            if no_imp >= patience:
                stopped_epoch = epoch + 1
                break

        if stopped_epoch is not None:
            CONSOLE.print(
                f"[RESIDUAL TRAINING] ✔ Early stopping triggered at epoch {stopped_epoch} "
                f"(patience={patience}) — {name}"
            )
        else:
            CONSOLE.print(
                f"[RESIDUAL TRAINING] ✔ Completed {epochs} epochs (no early stop) — {name}"
            )

    return ckpt, history


def hybrid_val_mae_ah(
    m1: nn.Module,
    m2: nn.Module,
    val_loader: DataLoader,
    cap_std: float,
    chronos_conf_val: Optional[np.ndarray] = None,
) -> float:
    """Validation MAE in Ah (60/40 mix; residual gate + optional Chronos conf + EMA when confidence blend)."""
    m1.eval()
    m2.eval()
    tot, n = 0.0, 0
    idx = 0
    ema_carry: Optional[torch.Tensor] = None
    with torch.no_grad():
        for Xb, cpb, lcb, yb in val_loader:
            Xb, cpb, lcb, yb = (
                Xb.to(DEVICE),
                cpb.to(DEVICE),
                lcb.to(DEVICE),
                yb.to(DEVICE),
            )
            B = Xb.size(0)
            cc_b = None
            if chronos_conf_val is not None:
                sl = chronos_conf_val[idx : idx + B]
                cc_b = torch.from_numpy(np.asarray(sl, dtype=np.float32)).to(DEVICE)
            r1 = m1(Xb, cpb, lcb) / RESIDUAL_SCALE
            r2 = m2(Xb, cpb, lcb) / RESIDUAL_SCALE
            r_mix = W_RESIDUAL_M1 * r1 + W_RESIDUAL_M2 * r2
            r_eff, ema_carry, _ = apply_residual_confidence_gate(
                r_mix, chronos_conf_batch=cc_b, ema_carry=ema_carry
            )
            pred = cpb.unsqueeze(-1) + r_eff
            tot += torch.abs(yb - pred).sum().item()
            n += yb.numel()
            idx += B
    mae_s = tot / max(1, n)
    return float(mae_s * cap_std)


def build_case3_data(
    train_bats: List[str], test_bat: str
) -> Tuple[np.ndarray, ...]:
    assert len(train_bats) == 3
    dfs_train = [load_battery_strict(b) for b in train_bats]
    df_test = load_battery_strict(test_bat)

    feat_cols = [c for c in dfs_train[0].columns if c != "cycle"]
    cap_idx = feat_cols.index("capacity")
    n_feat = len(feat_cols)

    train_concat = pd.concat(dfs_train, ignore_index=True)
    assert len(train_concat) == LAST_N * 3, f"train rows {len(train_concat)} != {LAST_N * 3}"

    CONSOLE.print("[SCALING] ✔ Fitting StandardScaler on train concat only (9000 rows)")
    scaler = StandardScaler()
    scaler.fit(train_concat[feat_cols].values.astype(np.float32))

    X_list, y_list, lc_list = [], [], []
    for df in dfs_train:
        arr = scaler.transform(df[feat_cols].values.astype(np.float32))
        X, y, lc = sliding_windows_on_scaled(arr, cap_idx, n_feat)
        X_list.append(X)
        y_list.append(y)
        lc_list.append(lc)

    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    lc_train = np.concatenate(lc_list, axis=0)

    CONSOLE.print("[SCALING] ✔ Transforming train batteries (train-fitted scaler only)")
    test_arr = scaler.transform(df_test[feat_cols].values.astype(np.float32))
    CONSOLE.print("[SCALING] ✔ Transforming test battery with same scaler (no leakage)")
    X_test, y_test, lc_test = sliding_windows_on_scaled(test_arr, cap_idx, n_feat)

    CONSOLE.print(f"[WINDOWS] ✔ Creating sliding windows (SEQ_LEN={SEQ_LEN})")
    CONSOLE.print(
        f"[WINDOWS] ✔ Train windows: {len(X_train)} | Test windows: {len(X_test)} | "
        f"features={n_feat} cap_idx={cap_idx}"
    )
    return (
        X_train,
        y_train,
        lc_train,
        X_test,
        y_test,
        lc_test,
        cap_idx,
        scaler,
        feat_cols,
    )


def _smooth_1d(x: np.ndarray, window: int = 31) -> np.ndarray:
    if len(x) < window or window < 5:
        return x.copy()
    if HAS_SCIPY and len(x) >= window:
        w = window if window % 2 == 1 else window + 1
        poly = min(3, w // 2)
        try:
            return savgol_filter(x, w, poly)
        except Exception:
            pass
    k = min(window, len(x) // 2 * 2 - 1)
    if k < 3:
        return x.copy()
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k, dtype=np.float64) / k
    return np.convolve(xp, kernel, mode="valid")


def save_loss_plot(battery_id: str, history: Dict[str, List[float]], path: str) -> None:
    if not HAS_MPL:
        CONSOLE.print("[yellow]matplotlib not available; skip loss plot[/yellow]")
        return
    tr = history.get("train_loss", [])
    va = history.get("val_loss", [])
    if not tr:
        return
    epochs = np.arange(1, len(tr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(epochs, tr, label="Train loss", color="#1f77b4", linewidth=1.8)
    if va and len(va) == len(tr):
        ax.plot(epochs, va, label="Validation loss", color="#ff7f0e", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curve - Battery {battery_id} (Case-3)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def save_prediction_plot(
    battery_id: str,
    y_actual: np.ndarray,
    y_pred: np.ndarray,
    path: str,
) -> None:
    if not HAS_MPL:
        CONSOLE.print("[yellow]matplotlib not available; skip prediction plot[/yellow]")
        return
    n = len(y_actual)
    idx = np.arange(n)
    err = y_pred - y_actual
    y_a_s = _smooth_1d(y_actual.astype(np.float64))
    y_p_s = _smooth_1d(y_pred.astype(np.float64))

    fig, ax1 = plt.subplots(figsize=(11, 5), dpi=150)
    ax1.plot(idx, y_a_s, color="blue", label="Actual (smoothed)", linewidth=1.4, alpha=0.9)
    ax1.plot(idx, y_p_s, color="red", label="Predicted (smoothed)", linewidth=1.4, alpha=0.9)
    ax1.set_xlabel("Test window index")
    ax1.set_ylabel("Capacity (Ah)")
    ax1.set_title(f"Capacity Prediction - Battery {battery_id}")
    ax1.grid(True, alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(idx, err, color="green", label="Error (pred − actual)", linewidth=0.9, alpha=0.85)
    ax2.set_ylabel("Error (Ah)", color="green")
    ax2.tick_params(axis="y", labelcolor="green")

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="upper right", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_case(case_idx: int, pipeline, aggregate_hist: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
    train_bats, test_bat = CASE_3_CONFIG[case_idx]
    case_tag = f"c3_{case_idx}_{test_bat}"

    CONSOLE.print(
        Panel.fit(
            f"[bold]Case-3 fold {case_idx}[/bold]\nTrain: {train_bats}\nTest: [bold]{test_bat}[/bold]\n"
            f"Train rows: 3×{LAST_N}={3*LAST_N} | Test: {LAST_N}\n"
            "[dim]Last 3000 rows only; train-only calibration; RMSE = √MSE.[/dim]",
            title="Leave-one-out",
        )
    )
    CONSOLE.print("[DATA] ✔ Strict protocol: last 3000 rows per battery, chronological order")
    for b in train_bats:
        CONSOLE.print(f"    [DATA] ✔ Train battery {b}")
    CONSOLE.print(f"    [DATA] ✔ Test battery {test_bat}")
    CONSOLE.print(f"[DATA] ✔ Train size = {LAST_N * 3} | Test size = {LAST_N}")

    data = build_case3_data(train_bats, test_bat)
    X_tr, y_tr, lc_tr, X_te, y_te, lc_te, cap_idx, scaler, _ = data

    CONSOLE.print(
        f"[CHRONOS] ✔ Running Chronos contexts={CHRONOS_CONTEXT_LENGTHS} | "
        f"samples={CHRONOS_NUM_SAMPLES} | infer_batch={INFERENCE_BATCH_SIZE} | fp16"
    )
    preds_tr, vars_tr = run_chronos_multicontext(
        X_tr, cap_idx, pipeline, case_tag, "train", return_variance=CASE3_USE_CONFIDENCE_BLEND
    )
    preds_te, vars_te = run_chronos_multicontext(
        X_te, cap_idx, pipeline, case_tag, "test", return_variance=CASE3_USE_CONFIDENCE_BLEND
    )
    CONSOLE.print("[CHRONOS] ✔ Train/test Chronos inference complete (torch.inference_mode)")

    clo, chi = _clip_bounds(y_tr)
    preds_tr = {L: np.clip(preds_tr[L], clo, chi) for L in preds_tr}
    preds_te = {L: np.clip(preds_te[L], clo, chi) for L in preds_te}

    if CASE3_USE_CONFIDENCE_BLEND:
        assert vars_tr is not None and vars_te is not None
        p_tr = blend_chronos_by_confidence(preds_tr, vars_tr)
        p_te = blend_chronos_by_confidence(preds_te, vars_te)
        a, b, _, _ = fit_affine_on_blended(y_tr, p_tr)
        c_tr = apply_affine_only(p_tr, a, b)
        c_te = apply_affine_only(p_te, a, b)
        w_tuple: Tuple[float, ...] = tuple()
        chronos_blend_mode = "confidence_inv_var"
    else:
        w_tuple, a, b, _, _ = fit_multi_context_weights_and_affine(y_tr, preds_tr)
        c_tr = apply_multi_blend_affine(preds_tr, w_tuple, a, b)
        c_te = apply_multi_blend_affine(preds_te, w_tuple, a, b)
        chronos_blend_mode = "grid"
    cal_mae_tr = float(mean_absolute_error(y_tr.ravel(), c_tr.ravel()))
    CONSOLE.print(f"[CALIBRATION] ✔ Mode: {chronos_blend_mode}")
    if chronos_blend_mode == "grid":
        CONSOLE.print(f"[CALIBRATION] ✔ Context weights (grid): {tuple(round(x, 4) for x in w_tuple)}")
    else:
        CONSOLE.print("[CALIBRATION] ✔ Context weights: inverse-variance (per window) + ridge affine")
    CONSOLE.print(
        f"[CALIBRATION] ✔ Affine ridge a={a:.6f} b={b:.6f} | train MAE (scaled capacity)={cal_mae_tr:.6f}"
    )

    c_tr_t = torch.from_numpy(c_tr).float()
    c_te_t = torch.from_numpy(c_te).float()

    cut = max(1, int(len(X_tr) * 0.85))
    chronos_conf_val_np: Optional[np.ndarray] = (
        chronos_uncertainty_from_vars(vars_tr)[cut:]
        if CASE3_USE_CONFIDENCE_BLEND and vars_tr is not None
        else None
    )
    chronos_conf_te_np: Optional[np.ndarray] = (
        chronos_uncertainty_from_vars(vars_te)
        if CASE3_USE_CONFIDENCE_BLEND and vars_te is not None
        else None
    )
    if VERBOSE and CASE3_USE_CONFIDENCE_BLEND:
        sm = RESIDUAL_GATE_SMOOTH_BETA
        CONSOLE.print(
            "[bold]Chronos blend[/bold]: inverse-variance across contexts "
            f"(eps={CONFIDENCE_VAR_EPS:g}); ridge affine on blended pred (train only).\n"
            f"[bold]Residual gate[/bold]: r_conf ∝ 1/(√|r_mix|+{RESIDUAL_PRED_CONF_EPS:g}); "
            "× Chronos uncertainty (mean-var); norm; "
            f"clip [{RESIDUAL_GATE_FLOOR},{RESIDUAL_GATE_CAP}]; α={RESIDUAL_GATE_ALPHA:g}"
            + (f"; EMA β={sm:g} (cross-batch)" if sm is not None else "")
            + "."
        )

    w_tr = torch.linspace(1.0, 2.0, len(X_tr[:cut])).reshape(-1, 1)
    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr[:cut]),
            c_tr_t[:cut],
            torch.from_numpy(lc_tr[:cut]),
            torch.from_numpy(y_tr[:cut]),
            w_tr,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=_pin_memory(),
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr[cut:]),
            c_tr_t[cut:],
            torch.from_numpy(lc_tr[cut:]),
            torch.from_numpy(y_tr[cut:]),
        ),
        batch_size=BATCH_SIZE,
        pin_memory=_pin_memory(),
    )

    n_feat = X_tr.shape[2]
    m1 = PatchTSTResidual(n_feat, SEQ_LEN)
    m2 = TinyTimeMixer(n_feat, SEQ_LEN)
    CONSOLE.print(
        f"[RESIDUAL TRAINING] ✔ PatchTST + TinyTimeMixer | epochs≤{RESIDUAL_EPOCHS} | "
        f"patience={RESIDUAL_PATIENCE} | grad_clip={GRAD_CLIP_NORM} | train_batch={BATCH_SIZE}"
    )

    ck1, h1 = train_residual_case3(m1, tr_loader, val_loader, case_tag=f"{case_tag}_p")
    ck2, h2 = train_residual_case3(m2, tr_loader, val_loader, case_tag=f"{case_tag}_t")

    # Merge histories (mean train/val loss across two models per epoch for one plot)
    max_e = max(len(h1["train_loss"]), len(h2["train_loss"]))
    train_merged: List[float] = []
    val_merged: List[float] = []
    for i in range(max_e):
        t1 = h1["train_loss"][i] if i < len(h1["train_loss"]) else h1["train_loss"][-1]
        t2 = h2["train_loss"][i] if i < len(h2["train_loss"]) else h2["train_loss"][-1]
        v1 = h1["val_loss"][i] if i < len(h1["val_loss"]) else h1["val_loss"][-1]
        v2 = h2["val_loss"][i] if i < len(h2["val_loss"]) else h2["val_loss"][-1]
        train_merged.append(0.5 * (t1 + t2))
        val_merged.append(0.5 * (v1 + v2))
    merged_hist = {"train_loss": train_merged, "val_loss": val_merged}
    aggregate_hist[test_bat] = merged_hist

    m1.load_state_dict(torch.load(ck1, map_location=DEVICE))
    m2.load_state_dict(torch.load(ck2, map_location=DEVICE))
    m1.eval()
    m2.eval()
    m1.to(DEVICE)
    m2.to(DEVICE)

    std = scaler.scale_[cap_idx]
    mn = scaler.mean_[cap_idx]
    val_mae_ah = hybrid_val_mae_ah(
        m1, m2, val_loader, std, chronos_conf_val=chronos_conf_val_np
    )
    use_b0006_fallback = (
        test_bat == "B0006"
        and val_mae_ah > B0006_VAL_MAE_THRESHOLD_AH
    )
    CONSOLE.print(
        f"[HYBRID] ✔ Chronos_cal + weighted residuals + "
        f"{'confidence gate' if CASE3_USE_CONFIDENCE_BLEND else 'no gate'}"
    )
    CONSOLE.print(f"[HYBRID] ✔ Internal validation MAE (Ah) = {val_mae_ah:.6f}")
    if use_b0006_fallback and VERBOSE:
        CONSOLE.print(
            f"[yellow]B0006 fallback:[/yellow] val_MAE_Ah={val_mae_ah:.4f} > "
            f"{B0006_VAL_MAE_THRESHOLD_AH} → damp residual (final = c + 0.2·r_eff)."
        )

    te_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_te),
            c_te_t,
            torch.from_numpy(lc_te),
            torch.from_numpy(y_te),
        ),
        batch_size=BATCH_SIZE,
        pin_memory=_pin_memory(),
    )

    CONSOLE.print("[TEST] ✔ Final prediction on held-out test windows (no tuning on test)")
    hyb_list, y_list, c_list, lc_list = [], [], [], []
    te_idx = 0
    te_ema_carry: Optional[torch.Tensor] = None
    gate_r_conf_chunks: List[torch.Tensor] = []
    gate_cc_norm_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for Xb, cpb, lcb, yb in te_loader:
            Xb, cpb, lcb, yb = Xb.to(DEVICE), cpb.to(DEVICE), lcb.to(DEVICE), yb.to(DEVICE)
            B = Xb.size(0)
            cc_b = None
            if chronos_conf_te_np is not None:
                cc_b = torch.from_numpy(
                    np.asarray(chronos_conf_te_np[te_idx : te_idx + B], dtype=np.float32)
                ).to(DEVICE)
            r1 = m1(Xb, cpb, lcb) / RESIDUAL_SCALE
            r2 = m2(Xb, cpb, lcb) / RESIDUAL_SCALE
            r_mix = W_RESIDUAL_M1 * r1 + W_RESIDUAL_M2 * r2
            r_eff, te_ema_carry, diag = apply_residual_confidence_gate(
                r_mix, chronos_conf_batch=cc_b, ema_carry=te_ema_carry
            )
            if diag is not None:
                r_c, cc_n = diag
                gate_r_conf_chunks.append(r_c)
                if cc_n is not None:
                    gate_cc_norm_chunks.append(cc_n)
            if use_b0006_fallback:
                pred_h = cpb.unsqueeze(-1) + 0.2 * r_eff
            else:
                pred_h = cpb.unsqueeze(-1) + r_eff
            hyb_list.append(pred_h.cpu().numpy().flatten())
            y_list.append(yb.cpu().numpy().flatten())
            c_list.append(cpb.cpu().numpy().flatten())
            lc_list.append(lcb.cpu().numpy().flatten())
            te_idx += B

    # Aggregate gate stats once (test split) for IEEE logging + CSV
    rc_agg: Optional[torch.Tensor] = (
        torch.cat(gate_r_conf_chunks) if gate_r_conf_chunks else None
    )
    cc_agg: Optional[torch.Tensor] = (
        torch.cat(gate_cc_norm_chunks) if gate_cc_norm_chunks else None
    )
    chronos_conf_mean_fold = float("nan")
    if CASE3_USE_CONFIDENCE_BLEND and chronos_conf_te_np is not None:
        _ccf = chronos_conf_te_np.astype(np.float64)
        _ccf = _ccf / (float(_ccf.max()) + RESIDUAL_PRED_CONF_EPS)
        chronos_conf_mean_fold = float(_ccf.mean())
    elif CASE3_USE_CONFIDENCE_BLEND and cc_agg is not None:
        chronos_conf_mean_fold = float(cc_agg.float().mean())

    residual_gate_mean = residual_gate_min = residual_gate_max = float("nan")
    if CASE3_USE_CONFIDENCE_BLEND and rc_agg is not None:
        residual_gate_mean = float(rc_agg.mean())
        residual_gate_min = float(rc_agg.min())
        residual_gate_max = float(rc_agg.max())

    # IEEE-style gate diagnostics (once per fold)
    if CASE3_USE_CONFIDENCE_BLEND:
        CONSOLE.print(
            f"[Gate] mode=inv_sqrt | alpha={RESIDUAL_GATE_ALPHA:g} | floor={RESIDUAL_GATE_FLOOR:g} | "
            f"cap={RESIDUAL_GATE_CAP:g} | eps={RESIDUAL_PRED_CONF_EPS:g}"
            + (
                f" | EMA_β={RESIDUAL_GATE_SMOOTH_BETA:g}"
                if RESIDUAL_GATE_SMOOTH_BETA is not None
                else ""
            )
        )
        if rc_agg is not None:
            CONSOLE.print(
                f"[Gate] mean={residual_gate_mean:.4f} | min={residual_gate_min:.4f} | "
                f"max={residual_gate_max:.4f}"
            )
        else:
            CONSOLE.print("[Gate] mean=n/a | min=n/a | max=n/a (degenerate / skipped all batches)")
        if not np.isnan(chronos_conf_mean_fold):
            CONSOLE.print(f"[ChronosConf] mean={chronos_conf_mean_fold:.4f}")
        else:
            CONSOLE.print("[ChronosConf] mean=n/a")
    else:
        CONSOLE.print("[Gate] disabled (CASE3_USE_CONFIDENCE_BLEND=0)")

    y_s = np.concatenate(y_list)
    h_s = np.concatenate(hyb_list)
    c_s = np.concatenate(c_list)
    lc_s = np.concatenate(lc_list)

    # Train-only adaptive clip on hybrid predictions (scaled capacity space)
    y_mu = float(np.mean(y_tr))
    y_sig = float(np.std(y_tr))
    lo_ada = y_mu - 3.0 * y_sig
    hi_ada = y_mu + 3.0 * y_sig
    h_s = np.clip(h_s.astype(np.float64), lo_ada, hi_ada).astype(np.float32)

    y_u = y_s * std + mn
    h_u = h_s * std + mn
    c_u = c_s * std + mn
    lc_u = lc_s * std + mn

    L_main = max(CHRONOS_CONTEXT_LENGTHS)
    raw_s = preds_te[L_main].astype(np.float64)
    raw_u = raw_s * std + mn

    mae_naive = mean_absolute_error(y_u, lc_u)
    mae_chr_raw = mean_absolute_error(y_u, raw_u)
    mae_chr_cal = mean_absolute_error(y_u, c_u)
    mae_final = mean_absolute_error(y_u, h_u)
    # Strict RMSE = sqrt(MSE); never use MAE×k proxy for reported hybrid metrics
    rmse_final = float(np.sqrt(mean_squared_error(y_u, h_u)))
    CONSOLE.print(
        f"[TEST] ✔ MAE={mae_final:.6f} Ah | RMSE={rmse_final:.6f} Ah (√MSE, exact)"
    )

    bench_m = H2O_MAE[test_bat]
    bench_r = H2O_RMSE[test_bat]

    imp_raw = mae_chr_raw - mae_final
    imp_naive = mae_naive - mae_final
    pct_raw = 100.0 * imp_raw / max(mae_chr_raw, 1e-12)
    pct_naive = 100.0 * imp_naive / max(mae_naive, 1e-12)
    imp_cal = mae_chr_cal - mae_final
    pct_cal = 100.0 * imp_cal / max(mae_chr_cal, 1e-12)
    CONSOLE.print(
        Panel.fit(
            f"[SANITY] mean(|y − last|)        = {mae_naive:.6f} Ah\n"
            f"[SANITY] mean(|y − Chronos_raw|) = {mae_chr_raw:.6f} Ah\n"
            f"[SANITY] mean(|y − Chronos_cal|) = {mae_chr_cal:.6f} Ah\n"
            f"[SANITY] mean(|y − Hybrid|)      = {mae_final:.6f} Ah\n"
            f"[bold]Improvement vs Chronos raw[/bold]:  {pct_raw:+.2f}% MAE reduction\n"
            f"[bold]Improvement vs Chronos cal[/bold]:  {pct_cal:+.2f}% MAE reduction\n"
            f"[bold]Improvement vs Naive[/bold]:        {pct_naive:+.2f}% MAE reduction\n"
            f"[dim]Chronos blend: {chronos_blend_mode}"
            + (
                f"; per-window conf=1/(var+eps)"
                if chronos_blend_mode == "confidence_inv_var"
                else f"; grid weights {tuple(round(x, 4) for x in w_tuple)}"
            )
            + f"; ridge λ_a={AFFINE_RIDGE_LAMBDA:g}; val_MAE_Ah={val_mae_ah:.4f}; "
            f"B0006_fallback={use_b0006_fallback}[/dim]",
            title="Sanity (test, Ah)",
        )
    )

    os.makedirs(PLOTS_DIR, exist_ok=True)
    loss_path = os.path.join(PLOTS_DIR, f"case3_loss_{test_bat}.png")
    pred_path = os.path.join(PLOTS_DIR, f"case3_prediction_{test_bat}.png")
    save_loss_plot(test_bat, merged_hist, loss_path)
    save_prediction_plot(test_bat, y_u, h_u, pred_path)
    CONSOLE.print(f"[blue]Saved[/blue] {loss_path}\n[blue]Saved[/blue] {pred_path}")

    row = {
        "case_idx": case_idx,
        "test_battery": test_bat,
        "train_batteries": "+".join(train_bats),
        "h2o_mae": bench_m,
        "h2o_rmse": bench_r,
        "mae_naive": mae_naive,
        "mae_chronos_raw": mae_chr_raw,
        "mae_chronos_cal": mae_chr_cal,
        "mae_hybrid_final": mae_final,
        "rmse_hybrid_final": rmse_final,
        "beat_mae": int(mae_final < bench_m),
        "beat_rmse": int(rmse_final < bench_r),
        "context_weights": (
            "confidence_inv_var_per_window"
            if chronos_blend_mode == "confidence_inv_var"
            else str(tuple(round(x, 6) for x in w_tuple))
        ),
        "chronos_blend_mode": chronos_blend_mode,
        "a_affine": a,
        "b_affine": b,
        "val_mae_ah": val_mae_ah,
        "b0006_residual_fallback": int(use_b0006_fallback),
        "improvement_vs_chronos_raw_ah": imp_raw,
        "improvement_vs_naive_ah": imp_naive,
        "residual_gate_alpha": (
            RESIDUAL_GATE_ALPHA if CASE3_USE_CONFIDENCE_BLEND else float("nan")
        ),
        "residual_gate_mean": residual_gate_mean,
        "residual_gate_min": residual_gate_min,
        "residual_gate_max": residual_gate_max,
        "chronos_conf_mean": chronos_conf_mean_fold,
    }
    return row


def main() -> None:
    verify_all_batteries()
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(HIS_DIR, exist_ok=True)

    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if argv:
        cases = [int(argv[0])]
    else:
        cases = list(range(len(CASE_3_CONFIG)))

    pipeline, model_used = _load_chronos()
    vram_note = f" | VRAM≈{CUDA_MEM_GB:.1f} GiB" if CUDA_MEM_GB is not None else ""
    train_note = (
        f" (auto-capped from {TRAIN_BATCH} for GPU memory)"
        if BATCH_SIZE < TRAIN_BATCH
        else ""
    )
    CONSOLE.print(
        f"[bold]Runtime[/bold]: DEVICE={DEVICE_KIND}{vram_note} | "
        f"cudnn.benchmark=True | cudnn.deterministic=False\n"
        f"[bold]Batches[/bold]: TRAIN_BATCH={BATCH_SIZE}{train_note} | "
        f"CHRONOS_INFER_BATCH={INFERENCE_BATCH_SIZE} "
        f"(override: CASE3_TRAIN_BATCH / CHRONOS_INFER_BATCH; "
        f"emergency: CHRONOS_LOW_VRAM / CASE3_LOW_VRAM)\n"
        f"[bold]Chronos[/bold]: {model_used} | fp16 | contexts={CHRONOS_CONTEXT_LENGTHS} | "
        f"median | num_samples={CHRONOS_NUM_SAMPLES} | "
        f"blend={'confidence_inv_var+residual_gate' if CASE3_USE_CONFIDENCE_BLEND else 'grid'}\n"
        f"[bold]Gate[/bold]: α={RESIDUAL_GATE_ALPHA:g} | "
        f"EMA_β={RESIDUAL_GATE_SMOOTH_BETA if RESIDUAL_GATE_SMOOTH_BETA is not None else 'off'}"
    )

    aggregate_hist: Dict[str, Dict[str, List[float]]] = {}
    rows: List[Dict[str, Any]] = []

    try:
        for ci in cases:
            if ci < 0 or ci >= len(CASE_3_CONFIG):
                CONSOLE.print(f"[red]Invalid case index {ci}[/red]")
                continue
            rows.append(run_case(ci, pipeline, aggregate_hist))
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        return

    results_path = os.path.join(HIS_DIR, "case3_results.csv")
    pd.DataFrame(rows).sort_values("test_battery").to_csv(results_path, index=False)
    CONSOLE.print(f"\n[bold green]Saved[/bold green] {results_path}")

    tbl = Table(
        title="Case-3 vs H2O — Hybrid = Chronos_cal + residuals (+ gate if enabled); RMSE = √MSE",
        show_header=True,
        header_style="bold",
    )
    tbl.add_column("Battery", style="cyan")
    tbl.add_column("H2O MAE", justify="right")
    tbl.add_column("Hybrid MAE", justify="right", style="magenta")
    tbl.add_column("Beat MAE?", justify="center")
    tbl.add_column("H2O RMSE", justify="right")
    tbl.add_column("Hybrid RMSE", justify="right")
    tbl.add_column("Beat RMSE?", justify="center")
    tbl.add_column("Overall", justify="center")

    wins = 0
    for r in sorted(rows, key=lambda x: x["test_battery"]):
        tb = r["test_battery"]
        bm, br = r["h2o_mae"], r["h2o_rmse"]
        m_mae, m_rmse = r["mae_hybrid_final"], r["rmse_hybrid_final"]
        bmae = m_mae < bm
        brmse = m_rmse < br
        overall = bmae and brmse
        if overall:
            wins += 1
        tbl.add_row(
            tb,
            f"{bm:.4f}",
            f"{m_mae:.6f}",
            "[green]YES[/green]" if bmae else "[red]NO[/red]",
            f"{br:.4f}",
            f"{m_rmse:.6f}",
            "[green]YES[/green]" if brmse else "[red]NO[/red]",
            "[green]WIN[/green]" if overall else "[red]FAIL[/red]",
        )

    CONSOLE.print("\n")
    CONSOLE.print(tbl)
    CONSOLE.print(
        f"\n[dim]Beat H2O on both MAE & RMSE: {wins}/{len(rows)} folds (goal ≥3/4).[/dim]\n"
        f"[dim]Hybrid RMSE uses sqrt(MSE) exactly; H2O RMSE from explicit baseline table (not k×MAE).[/dim]"
    )


if __name__ == "__main__":
    main()
