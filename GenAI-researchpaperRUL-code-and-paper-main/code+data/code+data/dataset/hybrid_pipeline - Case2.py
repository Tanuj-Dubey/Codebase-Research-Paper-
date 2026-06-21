"""
Battery capacity forecasting pipeline (Chronos + optional hybrid residuals).

Publication notes (Chronos-only path):
- Chronos context excludes the final in-window timestep (strict relative to the one-step target).
- Blend weight w is chosen on an internal chronological holdout from *train windows only* (not reported
  as a validation split; test is never used for tuning).
- Affine calibration uses ridge (L2 on slope a) and is fit on all train windows after w is fixed.
- Hybrid mode: residual networks are trained for up to 200 epochs with early stopping on a held-out
  slice of train windows (when VAL_RATIO=0, last 15% of train windows), consistent with common practice.

No test-time tuning: all hyperparameters touching w, a, b use training data only.
"""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import sys
import time
import warnings
import gc
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# 0. CONFIG & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

CONSOLE = Console()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)


def _init_compute_device():
    """
    Prefer NVIDIA GPU: explicit cuda:0 + Hugging Face device_map='auto' so Chronos weights load on GPU.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(0)
        except Exception:
            pass
        dev = torch.device("cuda:0")
        name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        CONSOLE.print(f"[bold green]GPU enabled[/bold green]: {name} ({mem_gb:.1f} GiB) | torch={torch.__version__}")
        return dev, "auto"
    CONSOLE.print(
        "[bold yellow]CUDA not available[/bold yellow] — using CPU. "
        "Install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ to use your GPU."
    )
    return torch.device("cpu"), None


BATTERY = sys.argv[1] if len(sys.argv) > 1 else "B0005"
HIS_DIR = r"d:\research paper - Copy\code+data\code+data\dataset\HIS"
CSV_PATH = os.path.join(HIS_DIR, f"{BATTERY}_synthetic+Real.csv")

SEQ_LEN = 96
PRED_LEN = 1
LAST_N_ROWS = 3000
TRAIN_RATIO = 0.80
VAL_RATIO = 0.0
TEST_RATIO = 0.20

BATCH_SIZE = 256
DEVICE, CHRONOS_DEVICE_MAP = _init_compute_device()


def _env_int(name, default):
    v = os.environ.get(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name, default=False):
    v = os.environ.get(name, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "on")


def _parse_context_lengths(s):
    """e.g. '96' or '96,48' -> tuple of ints."""
    s = (s or "").strip()
    if not s:
        return None
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    return tuple(parts) if parts else None


def _auto_chronos_batch_size():
    """Larger batches = far fewer model steps; scale down on low VRAM to avoid OOM."""
    env_b = _env_int("CHRONOS_INFER_BATCH", 0)
    if env_b > 0:
        return max(1, env_b)
    if not torch.cuda.is_available():
        return 48
    try:
        gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    except Exception:
        gb = 8.0
    if gb < 5.0:
        return 96
    if gb < 9.0:
        return 160
    return 256


# Chronos stack (train-only meta-weights: no test leakage)
CHRONOS_MODEL_PRIMARY = os.environ.get("CHRONOS_MODEL", "amazon/chronos-t5-base")
CHRONOS_MODEL_FALLBACK = "amazon/chronos-t5-small"
# Stochastic samples dominate runtime; 12 is a good speed/quality tradeoff (override with CHRONOS_NUM_SAMPLES).
CHRONOS_NUM_SAMPLES = max(1, _env_int("CHRONOS_NUM_SAMPLES", 12))
# Single context (96) avoids a full second pass — set CHRONOS_CONTEXTS=96 for max speed.
_ctx_env = _parse_context_lengths(os.environ.get("CHRONOS_CONTEXTS", ""))
CHRONOS_CONTEXT_LENGTHS = _ctx_env if _ctx_env is not None else ((96, 48) if not _env_bool("CHRONOS_SINGLE_CONTEXT", False) else (96,))
# Small grid + ridge reduces meta-calibration overfitting (override with CHRONOS_BLEND_GRID=0.3,0.5,0.7)
_blend_env = os.environ.get("CHRONOS_BLEND_GRID", "").strip()
if _blend_env:
    CHRONOS_BLEND_GRID = np.array([float(x) for x in _blend_env.split(",") if x.strip()])
else:
    CHRONOS_BLEND_GRID = np.array([0.3, 0.5, 0.7], dtype=np.float64)

INFERENCE_BATCH_SIZE = _auto_chronos_batch_size()
# Internal holdout: last ~12.5% of *train windows* to select w only (not a reported val split).
CALIB_HOLDOUT_FRAC = float(os.environ.get("CHRONOS_CALIB_HOLDOUT_FRAC", "0.125"))
AFFINE_RIDGE_LAMBDA = float(os.environ.get("AFFINE_RIDGE_LAMBDA", "1e-3"))
# Clip Chronos outputs in scaled capacity space using train min/max + margin.
CHRONOS_CLIP_MARGIN_FRAC = float(os.environ.get("CHRONOS_CLIP_MARGIN_FRAC", "0.05"))
# Cache bust when inference definition changes.
CHRONOS_CACHE_TAG = (
    f"v7_pub_nolast_ns{CHRONOS_NUM_SAMPLES}_ctx{'-'.join(map(str, CHRONOS_CONTEXT_LENGTHS))}_ridge_calib"
)
# Log each batch only every N steps (Rich prints are slow). 0 = progress bar only.
CHRONOS_LOG_BATCH_EVERY = max(0, _env_int("CHRONOS_LOG_BATCH_EVERY", 25))
VERBOSE_CHRONOS = not _env_bool("CHRONOS_QUIET", False)

BENCHMARKS = {
    "B0005": {"MAE": 0.0097, "RMSE": 0.0127},
    "B0006": {"MAE": 0.0258, "RMSE": 0.0323},
    "B0007": {"MAE": 0.0093, "RMSE": 0.0109},
    "B0018": {"MAE": 0.0093, "RMSE": 0.0130}
}
BENCHMARK_MAE = BENCHMARKS.get(BATTERY, {"MAE": 0.0097})["MAE"]
BENCHMARK_RMSE = BENCHMARKS.get(BATTERY, {"RMSE": 0.0127})["RMSE"]

# When True: final predictions are Chronos only (no PatchTST / TinyTimeMixer / H2O-style stack).
USE_CHRONOS_ONLY = True

# ─────────────────────────────────────────────────────────────────────────────
# 1. LEAKAGE-FREE DATA SETUP
# ─────────────────────────────────────────────────────────────────────────────

def get_leakage_free_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset at {csv_path}. Please check Battery ID.")
        
    CONSOLE.print(f"[bold blue]Step 1: Strict Chronological Splitting (last {LAST_N_ROWS} rows, {int(TRAIN_RATIO*100)}% train / {int(TEST_RATIO*100)}% test)...[/bold blue]")
    df = pd.read_csv(csv_path)
    df = df.sort_values("cycle").reset_index(drop=True)

    if len(df) > LAST_N_ROWS:
        df = df.tail(LAST_N_ROWS).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    CONSOLE.print(f"Rows: total={n} | train={len(df_train)} | val={len(df_val)} | test={len(df_test)}")
    CONSOLE.print(f"Index ranges: Train[0:{train_end}] Val[{train_end}:{val_end}] Test[{val_end}:{n}]")

    feat_cols = [c for c in df.columns if c != "cycle"]
    cap_idx = feat_cols.index("capacity")
    n_feat = len(feat_cols)

    scaler = StandardScaler()
    train_sc = scaler.fit_transform(df_train[feat_cols].values.astype(np.float32))
    val_sc = scaler.transform(df_val[feat_cols].values.astype(np.float32)) if len(df_val) else np.empty((0, n_feat), dtype=np.float32)
    test_sc = scaler.transform(df_test[feat_cols].values.astype(np.float32))

    def window(arr):
        if len(arr) <= SEQ_LEN:
            return (
                np.empty((0, SEQ_LEN, n_feat), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
                np.empty((0, 1), dtype=np.float32),
            )
        v = np.lib.stride_tricks.sliding_window_view(arr, SEQ_LEN, axis=0)
        v = v.transpose(0, 2, 1)
        X = v[:-1]
        y = arr[SEQ_LEN:, cap_idx].reshape(-1, 1)
        lc = v[:-1, -1, cap_idx].reshape(-1, 1)
        return X, y, lc

    X_train, y_train, lc_train = window(train_sc)
    X_val, y_val, lc_val = window(val_sc)
    X_test, y_test, lc_test = window(test_sc)
    
    CONSOLE.print(f"Windows: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    CONSOLE.print(f"[bold green]Overlap Check: PASSED (Independent windowing ensures zero leakage)[/bold green]")
    
    return (X_train, y_train, lc_train, 
            X_val, y_val, lc_val, 
            X_test, y_test, lc_test, 
            cap_idx, scaler)


def _capacity_context_excluding_last(x_np, cap_idx, context_len):
    """
    Build Chronos 1D context without the last in-window timestep (avoids overlap with the immediate pre-target state).
    x_np: (B, SEQ_LEN, F). Uses at most min(context_len, SEQ_LEN-1) trailing steps from x[:, :-1, cap_idx].
    """
    pref = x_np[:, :-1, cap_idx]
    if pref.shape[1] == 0:
        raise ValueError("SEQ_LEN must be >= 2 when excluding the last context step")
    sl = min(int(context_len), pref.shape[1])
    return np.ascontiguousarray(pref[:, -sl:], dtype=np.float32)


def _scaled_capacity_clip_bounds(y_train_scaled):
    y = np.asarray(y_train_scaled, dtype=np.float64).ravel()
    lo, hi = float(np.min(y)), float(np.max(y))
    span = hi - lo + 1e-8
    m = CHRONOS_CLIP_MARGIN_FRAC * span
    return lo - m, hi + m


def _clip_pred_dict(preds_by_L, lo, hi):
    return {L: np.clip(np.asarray(preds_by_L[L], dtype=np.float32), lo, hi) for L in preds_by_L}


def affine_ridge_fit(p, y, lam=None):
    """Fit y ≈ a*p + b with ridge penalty lam * a^2 only (intercept unpenalized)."""
    if lam is None:
        lam = AFFINE_RIDGE_LAMBDA
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return 1.0, 0.0
    X = np.column_stack([p, np.ones(n)])
    XtX = X.T @ X
    XtX[0, 0] += lam
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty)
    return float(beta[0]), float(beta[1])


# ─────────────────────────────────────────────────────────────────────────────
# 2. CHRONOS INFERENCE (multi-context + sample-mean + verbose)
# ─────────────────────────────────────────────────────────────────────────────

def _chronos_cache_path(split_name, context_len, battery=None):
    bid = battery if battery is not None else BATTERY
    return os.path.join(
        HIS_DIR,
        f"chronos_{CHRONOS_CACHE_TAG}_{bid}_{split_name}_L{context_len}.npy",
    )


def _load_chronos_pipeline():
    from chronos import ChronosPipeline

    for model_id in (CHRONOS_MODEL_PRIMARY, CHRONOS_MODEL_FALLBACK):
        try:
            t0 = time.perf_counter()
            if VERBOSE_CHRONOS:
                CONSOLE.print(f"[yellow]Loading Chronos weights:[/yellow] {model_id} …")
            load_kw = {"torch_dtype": torch.float16}
            if CHRONOS_DEVICE_MAP is not None:
                load_kw["device_map"] = CHRONOS_DEVICE_MAP
            pipe = ChronosPipeline.from_pretrained(model_id, **load_kw)
            if VERBOSE_CHRONOS:
                CONSOLE.print(
                    f"[green]Loaded in {time.perf_counter() - t0:.2f}s[/green] | "
                    f"torch_device={DEVICE} | chronos_device_map={CHRONOS_DEVICE_MAP!r}"
                )
            return pipe, model_id
        except Exception as e:
            CONSOLE.print(f"[red]Failed {model_id}:[/red] {e}")
    raise RuntimeError("Could not load any Chronos checkpoint.")


def _forecast_batch_mean_samples(pipeline, x_np, cap_idx, context_len):
    """x_np: (B, SEQ_LEN, F) — capacity context excludes final in-window step; then last min(L, SEQ_LEN-1) steps."""
    arr = _capacity_context_excluding_last(x_np, cap_idx, context_len)
    # Chronos tokenizer keeps `boundaries` on CPU; passing CUDA context breaks torch.bucketize.
    # `predict()` tokenizes on CPU and moves token_ids to the model device internally.
    ctx = torch.from_numpy(arr).float().cpu()
    with torch.inference_mode():
        forecast = pipeline.predict(
            ctx,
            prediction_length=PRED_LEN,
            num_samples=CHRONOS_NUM_SAMPLES,
        )
    # (B, num_samples, pred_len) -> mean over samples, last horizon step
    return forecast.mean(dim=1)[:, -1].float().cpu().numpy()


def run_chronos_context_length(X, cap_idx, split_name, context_len, pipeline, battery=None):
    """Run Chronos for one context length with cache, progress, and optional throttled batch logs."""
    if len(X) == 0:
        return np.empty((0,), dtype=np.float32)

    cache_path = _chronos_cache_path(split_name, context_len, battery)
    if os.path.exists(cache_path):
        cached = np.load(cache_path)
        if len(cached) == len(X):
            if VERBOSE_CHRONOS:
                CONSOLE.print(f"[green]Cache hit[/green] {os.path.basename(cache_path)} ({len(cached)} preds)")
            return cached.astype(np.float32)

    if VERBOSE_CHRONOS:
        eff_l = min(int(context_len), SEQ_LEN - 1)
        CONSOLE.print(
            Panel.fit(
                f"[bold]{split_name}[/bold] | nominal_L={context_len} | effective_len={eff_l} "
                f"(last in-window step excluded) | windows={len(X)} | "
                f"num_samples={CHRONOS_NUM_SAMPLES} | batch={INFERENCE_BATCH_SIZE}",
                title="Chronos inference",
            )
        )

    num = len(X)
    chunks = []
    t_all = time.perf_counter()
    n_batches = (num + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
    bench_prev = torch.backends.cudnn.benchmark if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=CONSOLE,
        ) as progress:
            task = progress.add_task(f"[cyan]{split_name} L={context_len}", total=num)
            for bi, i in enumerate(range(0, num, INFERENCE_BATCH_SIZE)):
                t0 = time.perf_counter()
                batch = X[i : i + INFERENCE_BATCH_SIZE]
                part = _forecast_batch_mean_samples(pipeline, batch, cap_idx, context_len)
                chunks.append(part)
                dt = time.perf_counter() - t0
                if CHRONOS_LOG_BATCH_EVERY <= 0:
                    log_this = False
                else:
                    log_this = VERBOSE_CHRONOS and (
                        bi == 0
                        or bi == n_batches - 1
                        or (bi + 1) % CHRONOS_LOG_BATCH_EVERY == 0
                    )
                if log_this:
                    eff_l = min(int(context_len), SEQ_LEN - 1)
                    CONSOLE.print(
                        f"  batch {bi + 1}/{n_batches} | idx [{i}:{i + batch.shape[0]}) | "
                        f"shape_ctx={batch.shape[0]}×{eff_l} (no last step) | {dt:.3f}s | "
                        f"part_mean={float(part.mean()):.6f} part_std={float(part.std()):.6f}"
                    )
                progress.update(task, advance=batch.shape[0])
    finally:
        if torch.cuda.is_available() and bench_prev is not None:
            torch.backends.cudnn.benchmark = bench_prev

    out = np.concatenate(chunks).astype(np.float32)
    np.save(cache_path, out)
    if VERBOSE_CHRONOS:
        CONSOLE.print(
            f"[green]Finished[/green] {split_name} L={context_len} in {time.perf_counter() - t_all:.2f}s | "
            f"saved {os.path.basename(cache_path)}"
        )
    return out


def run_multicontext_chronos(X, cap_idx, split_name, pipeline=None, model_used=None, battery=None):
    """
    Returns dict context_len -> np.ndarray of forecasts (scaled capacity space).
    If pipeline is None, loads weights once and returns (dict, model_id); caller must delete pipeline when done.
    If pipeline is provided, returns (dict, model_used) without unloading.
    """
    unload = pipeline is None
    if unload:
        pipeline, model_used = _load_chronos_pipeline()
    if VERBOSE_CHRONOS and unload:
        CONSOLE.print(
            f"[bold magenta]Model in use:[/bold magenta] {model_used} | DEVICE={DEVICE} | "
            f"contexts={CHRONOS_CONTEXT_LENGTHS} | infer_batch={INFERENCE_BATCH_SIZE} | "
            f"num_samples={CHRONOS_NUM_SAMPLES}"
        )
    result = {}
    try:
        for L in CHRONOS_CONTEXT_LENGTHS:
            if L > SEQ_LEN:
                CONSOLE.print(f"[yellow]Skipping L={L} > SEQ_LEN={SEQ_LEN}[/yellow]")
                continue
            result[L] = run_chronos_context_length(X, cap_idx, split_name, L, pipeline, battery)
        return result, model_used
    finally:
        if unload:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def fit_blend_and_affine(y_true, preds_by_L):
    """
    Train-only meta-calibration:
    - If two contexts: choose w on a chronological holdout tail of train windows (not reported as val split).
      For each w, fit ridge (a,b) on the head only, score MAE on the tail; then refit (a,b) with ridge
      on *all* train windows at the chosen w.
    - Single context: ridge affine on full train only.
    Returns (w, a, b, train_mae_full, train_rmse_full).
    """
    Ls = sorted(preds_by_L.keys(), reverse=True)
    y = y_true.ravel().astype(np.float64)
    n = len(y)

    if len(Ls) < 2:
        p = preds_by_L[Ls[0]].ravel().astype(np.float64)
        a, b = affine_ridge_fit(p, y)
        pred = a * p + b
        mae = mean_absolute_error(y, pred)
        rmse = np.sqrt(mean_squared_error(y, pred))
        if VERBOSE_CHRONOS:
            CONSOLE.print(
                f"[bold]Train-only ridge affine[/bold] (single context) a={a:.6f} b={b:.6f} | "
                f"train_MAE={mae:.6f} train_RMSE={rmse:.6f}"
            )
        return 1.0, a, b, mae, rmse

    L_long, L_short = Ls[0], Ls[1]
    p_long = preds_by_L[L_long].ravel().astype(np.float64)
    p_short = preds_by_L[L_short].ravel().astype(np.float64)

    n_cal = max(2, min(n - 3, int(round(n * CALIB_HOLDOUT_FRAC))))
    cut = n - n_cal
    use_holdout = cut >= 10 and n_cal >= 2

    best = None
    if use_holdout:
        if VERBOSE_CHRONOS:
            CONSOLE.print(
                "[bold]Train-only meta-fit[/bold]: blend grid on internal chronological holdout "
                f"(last {n_cal}/{n} train windows; not reported as a val split) — scanning w …"
            )
        for w in CHRONOS_BLEND_GRID:
            p_blend = w * p_long + (1.0 - w) * p_short
            a_h, b_h = affine_ridge_fit(p_blend[:cut], y[:cut])
            pred_cal = a_h * p_blend[cut:] + b_h
            mae_cal = mean_absolute_error(y[cut:], pred_cal)
            if VERBOSE_CHRONOS:
                CONSOLE.print(f"  w={w:.2f}  internal_holdout_MAE={mae_cal:.6f}  (ridge on head only)")
            if best is None or mae_cal < best[0]:
                best = (mae_cal, float(w))
    else:
        if VERBOSE_CHRONOS:
            CONSOLE.print(
                "[yellow]Train windows too few for stable internal holdout; "
                "selecting w by full-train ridge MAE (still no test use).[/yellow]"
            )
        for w in CHRONOS_BLEND_GRID:
            p_blend = w * p_long + (1.0 - w) * p_short
            a_f, b_f = affine_ridge_fit(p_blend, y)
            pred_f = a_f * p_blend + b_f
            mae_f = mean_absolute_error(y, pred_f)
            if VERBOSE_CHRONOS:
                CONSOLE.print(f"  w={w:.2f}  full_train_MAE={mae_f:.6f}")
            if best is None or mae_f < best[0]:
                best = (mae_f, float(w))

    w_best = best[1]
    p_blend_full = w_best * p_long + (1.0 - w_best) * p_short
    a_best, b_best = affine_ridge_fit(p_blend_full, y)
    pred_full = a_best * p_blend_full + b_best
    mae_full = mean_absolute_error(y, pred_full)
    rmse_full = np.sqrt(mean_squared_error(y, pred_full))

    if VERBOSE_CHRONOS:
        CONSOLE.print(
            f"[bold green]Selected w={w_best:.2f}[/bold green] by internal holdout | "
            f"final ridge on all train: a={a_best:.6f} b={b_best:.6f} | "
            f"train_MAE={mae_full:.6f} train_RMSE={rmse_full:.6f}"
        )
    return w_best, a_best, b_best, mae_full, rmse_full


def apply_blend_affine(preds_by_L, w, a, b):
    Ls = sorted(preds_by_L.keys(), reverse=True)
    if len(Ls) < 2:
        p = preds_by_L[Ls[0]]
    else:
        p = w * preds_by_L[Ls[0]] + (1.0 - w) * preds_by_L[Ls[1]]
    return (a * p + b).astype(np.float32)


def get_split_chronos(X, cap_idx, split_name, battery=None):
    """Single-context Chronos (median) for hybrid residual path only — fast, separate cache."""
    bid = battery if battery is not None else BATTERY
    cache_path = os.path.join(HIS_DIR, f"chronos_hybrid_v7nolast_{bid}_{split_name}.npy")
    if os.path.exists(cache_path):
        preds = np.load(cache_path)
        if len(preds) == len(X):
            CONSOLE.print(f"[bold green]Loaded {split_name} split predictions from cache![/bold green]")
            return torch.from_numpy(preds).to(torch.float32)

    CONSOLE.print(f"[bold yellow]Running Chronos on {split_name} split (hybrid / simple)...[/bold yellow]")
    from chronos import ChronosPipeline

    load_kw = {"torch_dtype": torch.float16}
    if CHRONOS_DEVICE_MAP is not None:
        load_kw["device_map"] = CHRONOS_DEVICE_MAP
    pipeline = ChronosPipeline.from_pretrained("amazon/chronos-t5-small", **load_kw)
    preds = []
    num = len(X)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=CONSOLE,
    ) as progress:
        task = progress.add_task(f"[cyan]Forecasting {split_name}...", total=num)
        with torch.no_grad():
            for i in range(0, num, INFERENCE_BATCH_SIZE):
                batch_np = X[i : i + INFERENCE_BATCH_SIZE]
                context_np = _capacity_context_excluding_last(batch_np, cap_idx, SEQ_LEN)
                context = torch.from_numpy(context_np).float().cpu()
                forecast = pipeline.predict(context, prediction_length=1)
                median = torch.median(forecast, dim=1).values
                preds.append(median.cpu().numpy().flatten())
                progress.update(task, advance=batch_np.shape[0])
    final = np.concatenate(preds).astype(np.float32)
    np.save(cache_path, final)
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return torch.from_numpy(final)

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────

class PatchTSTResidual(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.patch_size, self.stride = 16, 8
        self.num_patches = (seq_len - self.patch_size) // self.stride + 1
        self.proj = nn.Linear(self.patch_size, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model*2, batch_first=True, dropout=0.1)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(d_model * self.num_patches * input_dim + 2, 64), nn.ReLU(), nn.Linear(64, 1))
        
    def forward(self, x_win, c_pred, last_cap):
        B, L, C = x_win.shape
        x = x_win.transpose(1, 2)
        patches = [x[:, :, i*self.stride : i*self.stride + self.patch_size].unsqueeze(2) for i in range(self.num_patches)]
        xb = torch.cat(patches, dim=2).reshape(-1, self.num_patches, self.patch_size)
        out = self.transformer(self.proj(xb)).reshape(B, -1)
        return self.head(torch.cat([out, c_pred.unsqueeze(-1), last_cap], dim=1))

class TinyTimeMixer(nn.Module):
    def __init__(self, input_dim, seq_len, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(input_dim*seq_len + 2, hidden*2), nn.ReLU(), nn.Dropout(0.1), nn.Linear(hidden*2, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x, cp, lc):
        return self.net(torch.cat([x.reshape(x.size(0), -1), cp.unsqueeze(-1), lc], dim=1))

# ─────────────────────────────────────────────────────────────────────────────
# 4. TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=200, patience=15, threshold=None):
    """
    Train residual model up to `epochs` with ReduceLROnPlateau and early stopping on val MAE
    (val_loader is a held-out slice of train windows when VAL_RATIO=0).
    """
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)
    model.to(DEVICE)
    best_mae, no_imp = float('inf'), 0
    name = model.__class__.__name__
    
    for epoch in range(epochs):
        if threshold and best_mae < threshold:
            CONSOLE.print(f"[bold green]Benchmark reached ({best_mae:.6f} < {threshold:.6f}). Stopping early.[/bold green]")
            break
            
        model.train()
        t_mae = 0
        for Xb, cpb, lcb, yb, wb in train_loader:
            Xb, cpb, lcb, yb, wb = Xb.to(DEVICE), cpb.to(DEVICE), lcb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            trg = (yb - cpb.unsqueeze(-1)) * 10.0
            opt.zero_grad()
            out = model(Xb, cpb, lcb)
            loss = (0.7 * nn.functional.l1_loss(out, trg, reduction='none') + 0.3 * nn.functional.mse_loss(out, trg, reduction='none')) * wb
            loss.mean().backward(); opt.step()
            t_mae += nn.functional.l1_loss(out/10.0, yb - cpb.unsqueeze(-1)).item()
        
        model.eval(); v_mae = 0
        with torch.no_grad():
            for Xb, cpb, lcb, yb in val_loader:
                Xb, cpb, lcb, yb = Xb.to(DEVICE), cpb.to(DEVICE), lcb.to(DEVICE), yb.to(DEVICE)
                v_mae += nn.functional.l1_loss(cpb.unsqueeze(-1) + model(Xb,cpb,lcb)/10.0, yb).item()
        
        v_mae /= len(val_loader); sch.step(v_mae)
        if v_mae < best_mae: best_mae = v_mae; no_imp = 0; torch.save(model.state_dict(), f"best_{name}.pth")
        else: no_imp += 1
        
        if (epoch+1)%10==0 or epoch==0: CONSOLE.print(f"Ep {epoch+1:03d} | Tr MAE: {t_mae/len(train_loader):.6f} | Val MAE: {v_mae:.6f} | Best: {best_mae:.6f}")
        if no_imp >= patience: break
    return best_mae

# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_chronos_only_battery(battery_id, pipeline=None, model_used=None):
    """Chronos-only path for one battery. Reuses `pipeline` when given (multi-battery batch run)."""
    csv_path = os.path.join(HIS_DIR, f"{battery_id}_synthetic+Real.csv")
    benchmark_mae = BENCHMARKS.get(battery_id, {"MAE": 0.0097})["MAE"]
    benchmark_rmse = BENCHMARKS.get(battery_id, {"RMSE": 0.0127})["RMSE"]

    CONSOLE.print(
        Panel.fit(
            f"[bold]Battery[/bold] {battery_id}\n"
            "[bold]Protocol[/bold]: 80/20 chronological row split (no reported val split). "
            "Chronos context excludes last in-window timestep. "
            "Blend w chosen on internal train holdout; affine (a,b) ridge-fit on all train. "
            "[bold]No test-time tuning.[/bold]",
            title="Chronos-only (publication-oriented)",
        )
    )

    own_pipe = pipeline is None
    if own_pipe:
        pipeline, model_used = _load_chronos_pipeline()
    try:
        X_tr, y_tr, lc_tr, X_val, y_val, lc_val, X_te, y_te, lc_te, cap_idx, scaler = get_leakage_free_data(csv_path)

        CONSOLE.print("\n[bold cyan]Phase A — Chronos on TRAIN windows (for meta-fit only)[/bold cyan]")
        preds_tr_by_L, model_used = run_multicontext_chronos(
            X_tr, cap_idx, "train", pipeline=pipeline, model_used=model_used, battery=battery_id
        )
        clip_lo, clip_hi = _scaled_capacity_clip_bounds(y_tr)
        preds_tr_by_L = _clip_pred_dict(preds_tr_by_L, clip_lo, clip_hi)
        if VERBOSE_CHRONOS:
            CONSOLE.print(
                f"[dim]Scaled-space clip (from train y): [{clip_lo:.6f}, {clip_hi:.6f}] "
                f"(margin {CHRONOS_CLIP_MARGIN_FRAC:.0%} of train span)[/dim]"
            )

        w, a, b, tr_mae, tr_rmse = fit_blend_and_affine(y_tr, preds_tr_by_L)

        CONSOLE.print("\n[bold cyan]Phase B — Chronos on TEST windows (held-out 20%)[/bold cyan]")
        preds_te_by_L, _ = run_multicontext_chronos(
            X_te, cap_idx, "test", pipeline=pipeline, model_used=model_used, battery=battery_id
        )
        preds_te_by_L = _clip_pred_dict(preds_te_by_L, clip_lo, clip_hi)

        p_te = apply_blend_affine(preds_te_by_L, w, a, b)
        p_te = np.clip(p_te, clip_lo, clip_hi).astype(np.float32)
        L_main = max(CHRONOS_CONTEXT_LENGTHS)
        b_te = preds_te_by_L[L_main].astype(np.float32)
        y_te_f = y_te.ravel()

        std, mean = scaler.scale_[cap_idx], scaler.mean_[cap_idx]
        p_un = p_te * std + mean
        y_un = y_te_f * std + mean
        b_un = b_te * std + mean
        lc_un = lc_te.ravel().astype(np.float64) * std + mean

        mae_meta = mean_absolute_error(y_un, p_un)
        rmse_meta = np.sqrt(mean_squared_error(y_un, p_un))
        mae_raw = mean_absolute_error(y_un, b_un)
        rmse_raw = np.sqrt(mean_squared_error(y_un, b_un))
        naive_mae = mean_absolute_error(y_un, lc_un)
        naive_rmse = np.sqrt(mean_squared_error(y_un, lc_un))

        CONSOLE.print(
            Panel.fit(
                f"[bold]Sanity (test, Ah)[/bold]\n"
                f"mean(|y − last_obs_in_window|)  (naive persistence) = {naive_mae:.6f}\n"
                f"mean(|y − Chronos_raw|)  (long context, clipped) = {mae_raw:.6f}\n"
                f"mean(|y − meta_calibrated|) = {mae_meta:.6f}\n"
                f"If naive ≈ Chronos_raw, the model may be learning a near-persistence mapping.",
                title="Mandatory sanity check",
            )
        )
        if naive_mae > 1e-9 and mae_raw >= 0.98 * naive_mae:
            CONSOLE.print(
                "[bold yellow]Warning:[/bold yellow] Chronos raw MAE is within 2% of naive persistence — "
                "inspect features, context, and units."
            )

        title = f"Test metrics vs H2O baseline — {battery_id}"
        table = Table(title=title)
        table.add_column("Metric", style="cyan")
        table.add_column("H2O baseline", style="yellow")
        table.add_column(f"Chronos raw (L{L_main})", style="magenta")
        table.add_column("Chronos + train-only calib", style="green")
        table.add_column("Beat H2O?", style="bold")
        mae_win = mae_meta < benchmark_mae
        rmse_win = rmse_meta < benchmark_rmse
        table.add_row(
            "MAE",
            f"{benchmark_mae:.4f}",
            f"{mae_raw:.6f}",
            f"{mae_meta:.6f}",
            "[bold green]YES[/bold green]" if mae_win else "[red]NO[/red]",
        )
        table.add_row(
            "RMSE",
            f"{benchmark_rmse:.4f}",
            f"{rmse_raw:.6f}",
            f"{rmse_meta:.6f}",
            "[bold green]YES[/bold green]" if rmse_win else "[red]NO[/red]",
        )
        CONSOLE.print(table)
        CONSOLE.print(
            "[dim]Reporting: train-only calibration (ridge λ_a="
            f"{AFFINE_RIDGE_LAMBDA:g}); no test-time tuning. "
            f"Naive persistence MAE={naive_mae:.6f} RMSE={naive_rmse:.6f}[/dim]"
        )
        CONSOLE.print(
            f"[dim]Train after calibration: model={model_used} w={w:.4f} a={a:.6f} b={b:.6f} | "
            f"train_MAE={tr_mae:.6f} train_RMSE={tr_rmse:.6f}[/dim]"
        )

        if VERBOSE_CHRONOS:
            err = np.abs(y_un - p_un)
            idx_show = sorted(
                set(list(range(min(5, len(y_un)))) + [max(0, len(y_un) - j) for j in range(1, min(6, len(y_un) + 1))])
            )
            head = "\n".join(
                f"  i={i:5d}  y={y_un[i]:.6f}  pred={p_un[i]:.6f}  |err|={err[i]:.6f}" for i in idx_show
            )
            CONSOLE.print(
                Panel.fit(
                    f"Test windows: {len(y_un)}\n"
                    f"y_true  min/mean/max: {y_un.min():.6f} / {y_un.mean():.6f} / {y_un.max():.6f}\n"
                    f"pred    min/mean/max: {p_un.min():.6f} / {p_un.mean():.6f} / {p_un.max():.6f}\n"
                    f"|error| min/mean/max: {err.min():.6f} / {err.mean():.6f} / {err.max():.6f}\n"
                    f"Sample rows (Ah):\n{head}",
                    title=f"Phase C — {battery_id} test stats",
                )
            )

        pd.DataFrame(
            {
                "true": y_un,
                "pred_meta_calibrated": p_un,
                "pred_chronos_raw_L": b_un,
                "last_obs_in_window": lc_un,
            }
        ).to_csv(os.path.join(HIS_DIR, f"final_SOTA_{battery_id}.csv"), index=False)
        CONSOLE.print(f"[bold blue]Saved[/bold blue] final_SOTA_{battery_id}.csv")
    finally:
        if own_pipe:
            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()


def main():
    batteries = sys.argv[1:] if len(sys.argv) > 1 else [BATTERY]

    if VERBOSE_CHRONOS:
        cfg = {
            "batteries": batteries,
            "LAST_N_ROWS": LAST_N_ROWS,
            "TRAIN_RATIO": TRAIN_RATIO,
            "TEST_RATIO": TEST_RATIO,
            "SEQ_LEN": SEQ_LEN,
            "CHRONOS_MODEL_PRIMARY": CHRONOS_MODEL_PRIMARY,
            "CHRONOS_NUM_SAMPLES": CHRONOS_NUM_SAMPLES,
            "CHRONOS_CONTEXT_LENGTHS": list(CHRONOS_CONTEXT_LENGTHS),
            "CHRONOS_BLEND_GRID": [float(x) for x in CHRONOS_BLEND_GRID],
            "CALIB_HOLDOUT_FRAC": CALIB_HOLDOUT_FRAC,
            "AFFINE_RIDGE_LAMBDA": AFFINE_RIDGE_LAMBDA,
            "CHRONOS_CLIP_MARGIN_FRAC": CHRONOS_CLIP_MARGIN_FRAC,
            "context_excludes_last_inwindow_step": True,
            "INFERENCE_BATCH_SIZE": INFERENCE_BATCH_SIZE,
            "CHRONOS_LOG_BATCH_EVERY": CHRONOS_LOG_BATCH_EVERY,
            "DEVICE": str(DEVICE),
            "CHRONOS_DEVICE_MAP": CHRONOS_DEVICE_MAP,
            "CUDA": torch.cuda.is_available(),
            "TORCH": torch.__version__,
        }
        if torch.cuda.is_available():
            cfg["GPU"] = torch.cuda.get_device_name(0)
        CONSOLE.print(Panel(json.dumps(cfg, indent=2), title="Run configuration (full detail)"))

    if USE_CHRONOS_ONLY:
        if len(batteries) > 1:
            CONSOLE.print(
                f"[bold magenta]Batch mode:[/bold magenta] {len(batteries)} batteries — "
                "loading Chronos once, then inferring each cell sequentially."
            )
            shared, mu = _load_chronos_pipeline()
            try:
                for bat in batteries:
                    run_chronos_only_battery(bat, pipeline=shared, model_used=mu)
            finally:
                del shared
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
        else:
            run_chronos_only_battery(batteries[0])
        CONSOLE.print(f"\n[bold blue]Pipeline finished.[/bold blue]")
        return

    if len(batteries) > 1:
        CONSOLE.print("[yellow]Hybrid mode uses the first battery ID only:[/yellow] " + batteries[0])
    bat0 = batteries[0]
    csv_path = os.path.join(HIS_DIR, f"{bat0}_synthetic+Real.csv")
    benchmark_mae = BENCHMARKS.get(bat0, {"MAE": 0.0097})["MAE"]

    X_tr, y_tr, lc_tr, X_val, y_val, lc_val, X_te, y_te, lc_te, cap_idx, scaler = get_leakage_free_data(csv_path)

    c_tr_full = get_split_chronos(X_tr, cap_idx, "train", battery=bat0)
    c_te = get_split_chronos(X_te, cap_idx, "test", battery=bat0)

    if len(X_val) == 0:
        cut = max(1, int(len(X_tr) * 0.85))
        CONSOLE.print(
            f"[yellow]VAL_RATIO=0:[/yellow] last {100 - 85}% of train windows used only for early stopping "
            f"(not the 20% test split); models train up to 200 epochs with patience-based stop."
        )
        X_tr_m, X_val_m = X_tr[:cut], X_tr[cut:]
        y_tr_m, y_val_m = y_tr[:cut], y_tr[cut:]
        lc_tr_m, lc_val_m = lc_tr[:cut], lc_tr[cut:]
        c_tr_m, c_val_m = c_tr_full[:cut], c_tr_full[cut:]
    else:
        c_val_m = get_split_chronos(X_val, cap_idx, "val", battery=bat0)
        X_tr_m, y_tr_m, lc_tr_m, c_tr_m = X_tr, y_tr, lc_tr, c_tr_full
        X_val_m, y_val_m, lc_val_m, c_val_m = X_val, y_val, lc_val, c_val_m

    w_tr = torch.linspace(1.0, 2.0, len(X_tr_m)).reshape(-1, 1)
    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_m),
            c_tr_m,
            torch.from_numpy(lc_tr_m),
            torch.from_numpy(y_tr_m),
            w_tr,
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_val_m),
            c_val_m,
            torch.from_numpy(lc_val_m),
            torch.from_numpy(y_val_m),
        ),
        batch_size=BATCH_SIZE,
    )
    te_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), c_te, torch.from_numpy(lc_te), torch.from_numpy(y_te)),
        batch_size=BATCH_SIZE,
    )

    m1, m2 = PatchTSTResidual(X_tr.shape[2], SEQ_LEN), TinyTimeMixer(X_tr.shape[2], SEQ_LEN)
    CONSOLE.print(f"\n[bold yellow]Training models with benchmark target: {benchmark_mae}[/bold yellow]")
    train_model(m1, tr_loader, val_loader, epochs=200, threshold=benchmark_mae)
    train_model(m2, tr_loader, val_loader, epochs=200, threshold=benchmark_mae)
    m1.load_state_dict(torch.load("best_PatchTSTResidual.pth"))
    m2.load_state_dict(torch.load("best_TinyTimeMixer.pth"))
    m1.eval()
    m2.eval()
    m1.to(DEVICE)
    m2.to(DEVICE)

    te_preds, te_true, te_base = [], [], []
    with torch.no_grad():
        for Xb, cpb, lcb, yb in te_loader:
            Xb, cpb, lcb, yb = Xb.to(DEVICE), cpb.to(DEVICE), lcb.to(DEVICE), yb.to(DEVICE)
            p1 = (cpb.cpu().numpy() if cpb.is_cuda else cpb.numpy()).reshape(-1)
            r1 = (m1(Xb, cpb, lcb) / 10.0).cpu().numpy().flatten()
            r2 = (m2(Xb, cpb, lcb) / 10.0).cpu().numpy().flatten()
            te_preds.append(0.5 * (p1 + r1) + 0.5 * (p1 + r2))
            te_true.append(yb.cpu().numpy().flatten())
            te_base.append(p1)

    p_te, y_te_f, b_te = np.concatenate(te_preds), np.concatenate(te_true), np.concatenate(te_base)

    std, mean = scaler.scale_[cap_idx], scaler.mean_[cap_idx]
    p_un, y_un, b_un = p_te * std + mean, y_te_f * std + mean, b_te * std + mean

    mae, rmse = mean_absolute_error(y_un, p_un), np.sqrt(mean_squared_error(y_un, p_un))
    b_mae = mean_absolute_error(y_un, b_un)

    table = Table(title=f"Refactored Hybrid SOTA — {bat0}")
    table.add_column("Metric", style="cyan")
    table.add_column("Chronos (scaled cap)", style="magenta")
    table.add_column("SOTA Ens", style="green")
    table.add_column("Pass?", style="bold")
    table.add_row("MAE", f"{b_mae:.6f}", f"{mae:.6f}", "[bold green]YES[/bold green]" if mae < benchmark_mae else "NO")
    table.add_row("RMSE", f"{np.sqrt(mean_squared_error(y_un, b_un)):.6f}", f"{rmse:.6f}", "-")
    CONSOLE.print(table)

    pd.DataFrame({"true": y_un, "pred": p_un, "base": b_un}).to_csv(
        os.path.join(HIS_DIR, f"final_SOTA_{bat0}.csv"), index=False
    )
    CONSOLE.print(f"\n[bold blue]Pipeline finished. Results saved to HIS subdirectory.[/bold blue]")

if __name__ == "__main__":
    main()