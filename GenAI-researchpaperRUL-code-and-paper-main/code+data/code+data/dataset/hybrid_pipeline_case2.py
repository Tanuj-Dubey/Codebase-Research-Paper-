"""
Case-2 (2P1) cross-battery capacity forecasting: train on HI sequences from two batteries,
predict the held-out third battery. Strict leakage control:

- Exactly LAST_N rows per battery (assert len == LAST_N).
- Scaler fit ONLY on concatenated train batteries (6000 rows); test battery transformed with same scaler.
- Sliding windows built PER battery then merged — windows never cross battery boundaries.
- Chronos context = capacity[:, :-1] (last in-window timestep excluded).
- Blend + affine (ridge) fit on train windows only; internal chronological holdout selects blend w.
- Residual nets (PatchTST + TinyTimeMixer) trained on (y - Chronos_cal) on train only; early stopping
  on last 15% of train windows. Test battery never used for tuning.

No test-time tuning.
"""
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import time
import warnings
import gc
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

warnings.filterwarnings("ignore", category=UserWarning)

CONSOLE = Console()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HIS_DIR = os.path.join(SCRIPT_DIR, "HIS")

LAST_N = 3000
SEQ_LEN = 96
PRED_LEN = 1
BATCH_SIZE = 256
ALL_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]

CASE_2_CONFIG = [
    (["B0005", "B0006"], "B0007"),
    (["B0005", "B0007"], "B0018"),
    (["B0007", "B0006"], "B0005"),
    (["B0007", "B0018"], "B0006"),
]

# H2O-style targets (MAE). RMSE benchmarks approximated when not in spec (ratio ~1.28× MAE).
H2O_MAE = {
    "B0005": 0.0107,
    "B0006": 0.0115,
    "B0007": 0.0084,
    "B0018": 0.0128,
}
H2O_RMSE = {k: round(v * 1.28, 4) for k, v in H2O_MAE.items()}

CHRONOS_CONTEXT_LENGTHS = (96, 48)
CHRONOS_NUM_SAMPLES = max(1, int(os.environ.get("CHRONOS_NUM_SAMPLES", "12")))
CHRONOS_MODEL_PRIMARY = os.environ.get("CHRONOS_MODEL", "amazon/chronos-t5-base")
CHRONOS_MODEL_FALLBACK = "amazon/chronos-t5-small"
CHRONOS_BLEND_GRID = np.array([0.3, 0.5, 0.7], dtype=np.float64)
CALIB_HOLDOUT_FRAC = float(os.environ.get("CHRONOS_CALIB_HOLDOUT_FRAC", "0.125"))
AFFINE_RIDGE_LAMBDA = float(os.environ.get("AFFINE_RIDGE_LAMBDA", "1e-3"))
CHRONOS_CLIP_MARGIN_FRAC = float(os.environ.get("CHRONOS_CLIP_MARGIN_FRAC", "0.05"))
CHRONOS_LOG_BATCH_EVERY = max(0, int(os.environ.get("CHRONOS_LOG_BATCH_EVERY", "25")))
VERBOSE = os.environ.get("CHRONOS_QUIET", "").lower() not in ("1", "true", "yes")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)


def _init_device():
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


def _infer_batch():
    e = int(os.environ.get("CHRONOS_INFER_BATCH", "0") or 0)
    if e > 0:
        return max(1, e)
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


INFERENCE_BATCH_SIZE = _infer_batch()


def load_battery_strict(bat: str) -> pd.DataFrame:
    path = os.path.join(HIS_DIR, f"{bat}_synthetic+Real.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df.sort_values("cycle").reset_index(drop=True)
    df = df.tail(LAST_N).reset_index(drop=True)
    if len(df) != LAST_N:
        raise ValueError(f"{bat}: expected {LAST_N} rows after tail, got {len(df)}")
    return df


def verify_all_batteries():
    for bat in ALL_BATTERIES:
        df = load_battery_strict(bat)
        assert len(df) == LAST_N, bat
    CONSOLE.print(f"[bold green]Consistency:[/bold green] all {len(ALL_BATTERIES)} batteries have exactly {LAST_N} rows.")


def sliding_windows_on_scaled(arr: np.ndarray, cap_idx: int, n_feat: int):
    """arr: (T, n_feat) scaled. Returns X, y, lc; no cross-battery calls."""
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


def _capacity_context_excluding_last(x_np, cap_idx, context_len):
    pref = x_np[:, :-1, cap_idx]
    sl = min(int(context_len), pref.shape[1])
    return np.ascontiguousarray(pref[:, -sl:], dtype=np.float32)


def affine_ridge_fit(p, y, lam=AFFINE_RIDGE_LAMBDA):
    p = np.asarray(p, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(p)
    if n == 0:
        return 1.0, 0.0
    X = np.column_stack([p, np.ones(n)])
    XtX = X.T @ X
    XtX[0, 0] += lam
    beta = np.linalg.solve(XtX, X.T @ y)
    return float(beta[0]), float(beta[1])


def _clip_bounds(y_train):
    y = np.asarray(y_train, dtype=np.float64).ravel()
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo + 1e-8
    m = CHRONOS_CLIP_MARGIN_FRAC * span
    return lo - m, hi + m


def apply_blend_affine(preds_by_L, w, a, b):
    Ls = sorted(preds_by_L.keys(), reverse=True)
    if len(Ls) < 2:
        p = preds_by_L[Ls[0]]
    else:
        p = w * preds_by_L[Ls[0]] + (1.0 - w) * preds_by_L[Ls[1]]
    return (a * p + b).astype(np.float32)


def fit_blend_and_affine(y_true, preds_by_L):
    Ls = sorted(preds_by_L.keys(), reverse=True)
    y = y_true.ravel().astype(np.float64)
    n = len(y)
    if len(Ls) < 2:
        p = preds_by_L[Ls[0]].ravel().astype(np.float64)
        a, b = affine_ridge_fit(p, y)
        pred = a * p + b
        return 1.0, a, b, mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))

    p_long = preds_by_L[Ls[0]].ravel().astype(np.float64)
    p_short = preds_by_L[Ls[1]].ravel().astype(np.float64)
    n_cal = max(2, min(n - 3, int(round(n * CALIB_HOLDOUT_FRAC))))
    cut = n - n_cal
    use_holdout = cut >= 10

    best = None
    if use_holdout:
        if VERBOSE:
            CONSOLE.print(
                f"[bold]Meta-calibration[/bold]: select w on last {n_cal}/{n} train windows (internal holdout)."
            )
        for w in CHRONOS_BLEND_GRID:
            pb = w * p_long + (1.0 - w) * p_short
            ah, bh = affine_ridge_fit(pb[:cut], y[:cut])
            mae_c = mean_absolute_error(y[cut:], ah * pb[cut:] + bh)
            if VERBOSE:
                CONSOLE.print(f"  w={w:.2f}  holdout_MAE={mae_c:.6f}")
            if best is None or mae_c < best[0]:
                best = (mae_c, float(w))
        w_best = best[1]
    else:
        if VERBOSE:
            CONSOLE.print("[yellow]Small train: select w by full-train MAE.[/yellow]")
        for w in CHRONOS_BLEND_GRID:
            pb = w * p_long + (1.0 - w) * p_short
            a, b = affine_ridge_fit(pb, y)
            mae_f = mean_absolute_error(y, a * pb + b)
            if best is None or mae_f < best[0]:
                best = (mae_f, float(w))
        w_best = best[1]

    pb_full = w_best * p_long + (1.0 - w_best) * p_short
    a_best, b_best = affine_ridge_fit(pb_full, y)
    pred = a_best * pb_full + b_best
    return w_best, a_best, b_best, mean_absolute_error(y, pred), np.sqrt(mean_squared_error(y, pred))


def _load_chronos():
    from chronos import ChronosPipeline

    for mid in (CHRONOS_MODEL_PRIMARY, CHRONOS_MODEL_FALLBACK):
        try:
            kw = {"torch_dtype": torch.float16}
            if CHRONOS_DEVICE_MAP is not None:
                kw["device_map"] = CHRONOS_DEVICE_MAP
            return ChronosPipeline.from_pretrained(mid, **kw), mid
        except Exception as e:
            CONSOLE.print(f"[red]Chronos load failed {mid}:[/red] {e}")
    raise RuntimeError("No Chronos checkpoint loaded.")


def _forecast_batch_median(pipeline, x_np, cap_idx, context_len):
    arr = _capacity_context_excluding_last(x_np, cap_idx, context_len)
    ctx = torch.from_numpy(arr).float().cpu()
    with torch.inference_mode():
        fc = pipeline.predict(ctx, prediction_length=PRED_LEN, num_samples=CHRONOS_NUM_SAMPLES)
    med = torch.median(fc, dim=1).values[:, -1].float().cpu().numpy()
    return med


def run_chronos_multicontext(X, cap_idx, pipeline, case_tag: str, split_label: str):
    """No disk cache by default for Case-2 (case_tag unique); optional simple cache path."""
    out = {}
    for L in CHRONOS_CONTEXT_LENGTHS:
        if L > SEQ_LEN:
            continue
        n = len(X)
        parts = []
        n_b = (n + INFERENCE_BATCH_SIZE - 1) // INFERENCE_BATCH_SIZE
        bench = torch.backends.cudnn.benchmark if torch.cuda.is_available() else None
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=CONSOLE,
            ) as prog:
                task = prog.add_task(f"{case_tag} {split_label} L={L}", total=n)
                for bi, i in enumerate(range(0, n, INFERENCE_BATCH_SIZE)):
                    batch = X[i : i + INFERENCE_BATCH_SIZE]
                    parts.append(_forecast_batch_median(pipeline, batch, cap_idx, L))
                    prog.update(task, advance=batch.shape[0])
                    if VERBOSE and CHRONOS_LOG_BATCH_EVERY and (
                        bi == 0 or bi == n_b - 1 or (bi + 1) % CHRONOS_LOG_BATCH_EVERY == 0
                    ):
                        CONSOLE.print(f"  … batch {bi+1}/{n_b}")
        finally:
            if torch.cuda.is_available() and bench is not None:
                torch.backends.cudnn.benchmark = bench
        out[L] = np.concatenate(parts).astype(np.float32)
    return out


class PatchTSTResidual(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, n_heads=4, n_layers=2):
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
            nn.Linear(d_model * self.num_patches * input_dim + 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x_win, c_pred, last_cap):
        B, L, C = x_win.shape
        x = x_win.transpose(1, 2)
        patches = [
            x[:, :, i * self.stride : i * self.stride + self.patch_size].unsqueeze(2)
            for i in range(self.num_patches)
        ]
        xb = torch.cat(patches, dim=2).reshape(-1, self.num_patches, self.patch_size)
        out = self.transformer(self.proj(xb)).reshape(B, -1)
        return self.head(torch.cat([out, c_pred.unsqueeze(-1), last_cap], dim=1))


class TinyTimeMixer(nn.Module):
    def __init__(self, input_dim, seq_len, hidden=64):
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

    def forward(self, x, cp, lc):
        return self.net(torch.cat([x.reshape(x.size(0), -1), cp.unsqueeze(-1), lc], dim=1))


def train_residual(model, train_loader, val_loader, epochs=200, patience=15, case_tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)
    model.to(DEVICE)
    best_mae, no_imp = float("inf"), 0
    name = model.__class__.__name__
    ckpt = os.path.join(SCRIPT_DIR, f"best_{name}_{case_tag}.pth")

    for epoch in range(epochs):
        model.train()
        t_mae = 0.0
        for Xb, cpb, lcb, yb, wb in train_loader:
            Xb, cpb, lcb, yb, wb = (
                Xb.to(DEVICE),
                cpb.to(DEVICE),
                lcb.to(DEVICE),
                yb.to(DEVICE),
                wb.to(DEVICE),
            )
            trg = (yb - cpb.unsqueeze(-1)) * 10.0
            opt.zero_grad()
            out = model(Xb, cpb, lcb)
            loss = (
                0.7 * nn.functional.l1_loss(out, trg, reduction="none")
                + 0.3 * nn.functional.mse_loss(out, trg, reduction="none")
            ) * wb
            loss.mean().backward()
            opt.step()
            t_mae += nn.functional.l1_loss(out / 10.0, yb - cpb.unsqueeze(-1)).item()

        model.eval()
        v_mae = 0.0
        with torch.no_grad():
            for Xb, cpb, lcb, yb in val_loader:
                Xb, cpb, lcb, yb = (
                    Xb.to(DEVICE),
                    cpb.to(DEVICE),
                    lcb.to(DEVICE),
                    yb.to(DEVICE),
                )
                v_mae += nn.functional.l1_loss(
                    cpb.unsqueeze(-1) + model(Xb, cpb, lcb) / 10.0, yb
                ).item()
        v_mae /= max(1, len(val_loader))
        sch.step(v_mae)
        if v_mae < best_mae:
            best_mae = v_mae
            no_imp = 0
            torch.save(model.state_dict(), ckpt)
        else:
            no_imp += 1
        if VERBOSE and (epoch == 0 or (epoch + 1) % 20 == 0):
            CONSOLE.print(
                f"  {name} Ep {epoch+1:03d} | Tr MAE {t_mae/max(1,len(train_loader)):.6f} | Val {v_mae:.6f}"
            )
        if no_imp >= patience:
            break
    return ckpt


def build_case_data(train_bats, test_bat):
    """Scaler on concat train only; windows per battery; merge train windows."""
    dfs_train = [load_battery_strict(b) for b in train_bats]
    df_test = load_battery_strict(test_bat)

    feat_cols = [c for c in dfs_train[0].columns if c != "cycle"]
    cap_idx = feat_cols.index("capacity")
    n_feat = len(feat_cols)

    train_concat = pd.concat(dfs_train, ignore_index=True)
    assert len(train_concat) == LAST_N * 2, f"train rows {len(train_concat)} != {LAST_N*2}"

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

    test_arr = scaler.transform(df_test[feat_cols].values.astype(np.float32))
    X_test, y_test, lc_test = sliding_windows_on_scaled(test_arr, cap_idx, n_feat)

    CONSOLE.print(
        f"[blue]Windows[/blue]: train={len(X_train)} (no cross-battery) | test={len(X_test)} | "
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


def run_case(case_idx: int, pipeline):
    train_bats, test_bat = CASE_2_CONFIG[case_idx]
    case_tag = f"c{case_idx}_{'_'.join(train_bats)}_{test_bat}"

    CONSOLE.print(
        Panel.fit(
            f"[bold]Case {case_idx}[/bold]\nTrain: {train_bats}  →  Test: [bold]{test_bat}[/bold]\n"
            f"Rows: train batteries 2×{LAST_N}={2*LAST_N}; test {LAST_N}\n"
            "[dim]Scaler fit on train concat only. No test leakage.[/dim]",
            title="Case-2 (2P1)",
        )
    )

    data = build_case_data(train_bats, test_bat)
    X_tr, y_tr, lc_tr, X_te, y_te, lc_te, cap_idx, scaler, _ = data

    model_used = None
    preds_tr = run_chronos_multicontext(X_tr, cap_idx, pipeline, case_tag, "train")
    preds_te = run_chronos_multicontext(X_te, cap_idx, pipeline, case_tag, "test")

    clo, chi = _clip_bounds(y_tr)
    preds_tr = {L: np.clip(preds_tr[L], clo, chi) for L in preds_tr}
    preds_te = {L: np.clip(preds_te[L], clo, chi) for L in preds_te}

    w, a, b, _, _ = fit_blend_and_affine(y_tr, preds_tr)
    c_tr = apply_blend_affine(preds_tr, w, a, b)
    c_te = apply_blend_affine(preds_te, w, a, b)
    c_tr_t = torch.from_numpy(c_tr).float()
    c_te_t = torch.from_numpy(c_te).float()

    cut = max(1, int(len(X_tr) * 0.85))
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
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr[cut:]),
            c_tr_t[cut:],
            torch.from_numpy(lc_tr[cut:]),
            torch.from_numpy(y_tr[cut:]),
        ),
        batch_size=BATCH_SIZE,
    )

    n_feat = X_tr.shape[2]
    m1 = PatchTSTResidual(n_feat, SEQ_LEN)
    m2 = TinyTimeMixer(n_feat, SEQ_LEN)
    if VERBOSE:
        CONSOLE.print("[bold]Residual training[/bold] (up to 200 epochs, early stopping on internal val)…")
    ckpt_m1 = train_residual(m1, tr_loader, val_loader, case_tag=case_tag)
    ckpt_m2 = train_residual(m2, tr_loader, val_loader, case_tag=case_tag)
    m1.load_state_dict(torch.load(ckpt_m1, map_location=DEVICE))
    m2.load_state_dict(torch.load(ckpt_m2, map_location=DEVICE))
    m1.eval()
    m2.eval()
    m1.to(DEVICE)
    m2.to(DEVICE)

    std = scaler.scale_[cap_idx]
    mn = scaler.mean_[cap_idx]

    te_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_te),
            c_te_t,
            torch.from_numpy(lc_te),
            torch.from_numpy(y_te),
        ),
        batch_size=BATCH_SIZE,
    )

    hyb_list, y_list, c_list, lc_list = [], [], [], []
    with torch.no_grad():
        for Xb, cpb, lcb, yb in te_loader:
            Xb, cpb, lcb, yb = Xb.to(DEVICE), cpb.to(DEVICE), lcb.to(DEVICE), yb.to(DEVICE)
            r1 = m1(Xb, cpb, lcb) / 10.0
            r2 = m2(Xb, cpb, lcb) / 10.0
            pred_h = cpb.unsqueeze(-1) + 0.5 * (r1 + r2)
            hyb_list.append(pred_h.cpu().numpy().flatten())
            y_list.append(yb.cpu().numpy().flatten())
            c_list.append(cpb.cpu().numpy().flatten())
            lc_list.append(lcb.cpu().numpy().flatten())

    y_s = np.concatenate(y_list)
    h_s = np.concatenate(hyb_list)
    c_s = np.concatenate(c_list)
    lc_s = np.concatenate(lc_list)

    y_u = y_s * std + mn
    h_u = h_s * std + mn
    c_u = c_s * std + mn
    lc_u = lc_s * std + mn

    L_main = max(CHRONOS_CONTEXT_LENGTHS)
    raw_s = preds_te[L_main].astype(np.float64)
    raw_u = raw_s * std + mn

    mae_naive = mean_absolute_error(y_u, lc_u)
    mae_chr_raw = mean_absolute_error(y_u, raw_u)
    mae_chr = mean_absolute_error(y_u, c_u)
    mae_hyb = mean_absolute_error(y_u, h_u)
    rmse_hyb = np.sqrt(mean_squared_error(y_u, h_u))

    bench_m = H2O_MAE[test_bat]
    bench_r = H2O_RMSE[test_bat]

    table = Table(title=f"Case {case_idx} — Test {test_bat} vs H2O")
    table.add_column("Metric", style="cyan")
    table.add_column("Naive", style="white")
    table.add_column("Chronos raw", style="dim")
    table.add_column("Chronos cal.", style="magenta")
    table.add_column("Hybrid", style="green")
    table.add_column("H2O MAE", style="yellow")
    table.add_column("Beat?", style="bold")
    win_m = mae_hyb < bench_m
    win_r = rmse_hyb < bench_r
    table.add_row(
        "MAE",
        f"{mae_naive:.6f}",
        f"{mae_chr_raw:.6f}",
        f"{mae_chr:.6f}",
        f"{mae_hyb:.6f}",
        f"{bench_m:.4f}",
        "[green]YES[/green]" if win_m else "[red]NO[/red]",
    )
    table.add_row(
        "RMSE",
        f"{np.sqrt(mean_squared_error(y_u, lc_u)):.6f}",
        f"{np.sqrt(mean_squared_error(y_u, raw_u)):.6f}",
        f"{np.sqrt(mean_squared_error(y_u, c_u)):.6f}",
        f"{rmse_hyb:.6f}",
        f"{bench_r:.4f}",
        "[green]YES[/green]" if win_r else "[red]NO[/red]",
    )
    CONSOLE.print(table)

    CONSOLE.print(
        Panel.fit(
            f"mean(|y − last_value|)     = {mae_naive:.6f}\n"
            f"mean(|y − Chronos_raw|)    = {mae_chr_raw:.6f}  (L{L_main} median, clipped)\n"
            f"mean(|y − Chronos_cal|)    = {mae_chr:.6f}\n"
            f"mean(|y − hybrid|)         = {mae_hyb:.6f}\n"
            f"[dim]Train-only calibration (w={w:.2f}, ridge λ_a={AFFINE_RIDGE_LAMBDA:g}); "
            f"residuals on train only; no test leakage.[/dim]",
            title="Sanity summary (test, Ah)",
        )
    )

    out_csv = os.path.join(HIS_DIR, f"case2_{case_tag}_test_{test_bat}.csv")
    pd.DataFrame(
        {
            "true": y_u,
            "pred_hybrid": h_u,
            "pred_chronos_cal": c_u,
            "pred_chronos_raw_L": raw_u,
            "last_obs_window": lc_u,
        }
    ).to_csv(out_csv, index=False)
    CONSOLE.print(f"[blue]Saved[/blue] {out_csv}")
    return win_m and win_r, mae_hyb, rmse_hyb


def main():
    verify_all_batteries()
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    if argv:
        cases = [int(argv[0])]
    else:
        cases = list(range(len(CASE_2_CONFIG)))

    pipeline, model_used = _load_chronos()
    CONSOLE.print(f"[magenta]Chronos[/magenta]: {model_used} | contexts={CHRONOS_CONTEXT_LENGTHS} | median samples")

    results = []
    try:
        for ci in cases:
            if ci < 0 or ci >= len(CASE_2_CONFIG):
                CONSOLE.print(f"[red]Invalid case index {ci}[/red]")
                continue
            ok, mae, rmse = run_case(ci, pipeline)
            results.append((ci, ok, mae, rmse))
    finally:
        del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    CONSOLE.print("\n[bold]Case-2 summary[/bold]")
    for ci, ok, mae, rmse in results:
        tb = CASE_2_CONFIG[ci][1]
        CONSOLE.print(f"  case {ci} test {tb}: MAE={mae:.6f} RMSE={rmse:.6f}  all_metrics_beat_H2O={ok}")


if __name__ == "__main__":
    main()
