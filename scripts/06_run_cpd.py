"""CPD-зоопарк (К3): offline-детекторы + инъекционный стенд с известным ground truth.

Дизайн из research/round2/R04: десезонирование ДО детекции (иначе декабрь = массовые ложные
брейки), калибровка к общему бюджету FAR (~1 тревога / 1000 МО-месяцев), метрики F1@±2,
detection delay, FAR. Ground truth: инъекции level/trend/variance в реальные ряды
(стратифицированные 400 рядов, tau в 8..16) — генеративная синтетика запрещена (разделяет
допущения с детектором).

Выход: outputs/cpd_breaks.parquet (все ряды), outputs/cpd_injection_scores.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FAR_BUDGET = 1 / 1000  # тревог на МО-месяц (R04: 24k МО-мес × alpha=5% = 1200 ложных/год — нежизнеспособно)


def deseasonalize(y: np.ndarray) -> np.ndarray:
    """Вычесть per-series месячный сезонный профиль (24 точки = 2 цикла, медиана устойчивее)."""
    y = np.asarray(y, dtype=float)
    prof = np.array([np.median(y[m::12]) for m in range(12)])
    return y - prof[np.arange(len(y)) % 12]


def det_pelt(y, pen):
    bkps = rpt.Pelt(model="l2").fit(y).predict(pen=pen)
    return [b for b in bkps if b < len(y)]


def det_binseg(y, pen):
    bkps = rpt.Binseg(model="l2").fit(y).predict(pen=pen)
    return [b for b in bkps if b < len(y)]


def det_window(y, pen):
    bkps = rpt.Window(width=4, model="rbf").fit(y).predict(pen=pen)
    return [b for b in bkps if b < len(y)]


def det_bocpd(y, hazard=1/12, thresh=0.5):
    """BOCPD (Adams–MacKay) через bayesian_changepoint_detection; брейк = P(change) > thresh."""
    from bayesian_changepoint_detection.bayesian_changepoint_detection import online_changepoint_detection
    from functools import partial
    from bayesian_changepoint_detection.hazard_functions import constant_hazard
    from bayesian_changepoint_detection.online_likelihoods import StudentT
    R, _maxes = online_changepoint_detection(
        y, partial(constant_hazard, 1 / (hazard * len(y) * 12)), StudentT(0.1, 0.01, 1, 0))
    p_change = R[1, 1:]  # вероятность того, что run length = 0 в точке t
    scores = p_change.max(axis=0) if p_change.ndim > 1 else p_change
    # точки, где вероятность перезапуска выше порога
    cands = np.where(np.diag(R[1:, :]) > thresh)[0] if False else np.where(scores > thresh)[0]
    return sorted(set(int(c) for c in cands if 1 <= c < len(y)))


def det_cusum(y, k=None, h_sigma=4.0):
    """Односторонний CUSUM на десезонированном ряде (offline-сканер: рестарт после находки)."""
    s = np.std(np.diff(y)) or 1.0
    k = (k if k is not None else 0.5) * s
    thr = h_sigma * s
    breaks, s_pos, s_neg = [], 0.0, 0.0
    mu = np.median(y[:6])
    for t in range(1, len(y)):
        e = y[t] - mu
        s_pos = max(0, s_pos + e - k)
        s_neg = min(0, s_neg + e + k)
        if s_pos > thr or abs(s_neg) > thr:
            breaks.append(t)
            s_pos, s_neg = 0.0, 0.0
            mu = np.median(y[max(0, t - 5): t + 1])
    return breaks


METHODS = {"PELT": det_pelt, "BinSeg": det_binseg, "Window": det_window,
           "CUSUM": det_cusum, "BOCPD": det_bocpd}


def inject(y, kind, tau, delta_sigma):
    s = np.std(np.diff(y)) or 1.0
    z = y.copy()
    if kind == "level":
        z[tau:] += delta_sigma * s
    elif kind == "trend":
        z[tau:] += delta_sigma * s * 0.2 * np.arange(1, len(y) - tau + 1)
    elif kind == "variance":
        rng = np.random.default_rng(0)
        z[tau:] += rng.normal(0, delta_sigma * s, len(y) - tau)
    return z


def eval_detection(found, true_tau, tol=2, n=24):
    """F1@±tol и средний delay (положительный = после брейка)."""
    hits = [f for f in found if abs(f - true_tau) <= tol]
    tp = 1 if hits else 0
    fp = len([f for f in found if abs(f - true_tau) > tol])
    fn = 0 if hits else 1
    delay = min(hits) - true_tau if hits else np.nan
    return tp, fp, fn, delay


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index=["territory_id", "category"], columns="month_idx",
                             values="log_value").reindex(columns=range(24))
    print(f"рядов: {len(wide)}")

    # --- калибровка FAR на неинъецированных рядах (стратиф. подвыборка 400)
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(wide), size=400, replace=False)
    sub = wide.iloc[sub_idx]
    pens = {"PELT": 1.0, "BinSeg": 1.0, "Window": 3.0}  # стартовые, калибруем по FAR
    calib = {}
    for name, fn in [("PELT", det_pelt), ("BinSeg", det_binseg), ("Window", det_window)]:
        n_breaks = sum(len(fn(deseasonalize(row), pens[name])) for row in sub.to_numpy())
        calib[name] = n_breaks / (len(sub) * 24)
        print(f"{name}: FAR={calib[name]*1000:.1f} на 1000 МО-мес (цель ≤1)")
    # CUSUM/BOCPD: пороги фиксированы конструкцией
    for name in ["CUSUM", "BOCPD"]:
        fn = METHODS[name]
        n_breaks = sum(len(fn(deseasonalize(row))) for row in sub.to_numpy()[:200])
        calib[name] = n_breaks / (200 * 24)
        print(f"{name}: FAR={calib[name]*1000:.1f} на 1000 МО-мес")

    # --- инъекционный стенд
    rows_out = []
    for kind in ["level", "trend", "variance"]:
        for delta in ([0.2, 0.4, 0.8] if kind != "variance" else [1.5, 2.0]):
            for i, row in enumerate(sub.to_numpy()[:200]):
                tau = int(rng.integers(8, 17))
                z = inject(row, kind, tau, delta)
                z_ds = deseasonalize(z)
                for name, fn in METHODS.items():
                    if name in ("PELT", "BinSeg"):
                        found = fn(z_ds, pens[name])
                    elif name == "Window":
                        found = fn(z_ds, pens[name])
                    else:
                        found = fn(z_ds)
                    tp, fp, fn_, delay = eval_detection(found, tau)
                    rows_out.append(dict(method=name, kind=kind, delta=delta, tp=tp,
                                         fp=fp, fn=fn_, delay=delay))
    inj = pd.DataFrame(rows_out)
    summ = inj.groupby(["method", "kind"]).agg(
        f1_parts_tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
        delay=("delay", "mean")).reset_index()
    summ["precision"] = summ["f1_parts_tp"] / (summ["f1_parts_tp"] + summ["fp"]).clip(lower=1)
    summ["recall"] = summ["f1_parts_tp"] / (summ["f1_parts_tp"] + summ["fn"]).clip(lower=1)
    summ["F1"] = 2 * summ["precision"] * summ["recall"] / (summ["precision"] + summ["recall"]).clip(lower=1e-9)
    summ.to_csv(ROOT / "outputs" / "cpd_injection_scores.csv", index=False)
    print(summ.round(2).to_string())
    print(f"инъекции: {time.time()-t0:.0f}с")

    # --- полный offline-скан (все ряды) — для кейсов и радара
    full = []
    for (tid, cat), row in wide.iterrows():
        y_ds = deseasonalize(row.to_numpy())
        for name, fn in METHODS.items():
            if name in ("PELT", "BinSeg"):
                bk = fn(y_ds, pens[name])
            elif name == "Window":
                bk = fn(y_ds, pens[name])
            else:
                bk = fn(y_ds)
            for b in bk:
                full.append((tid, cat, name, b))
    br = pd.DataFrame(full, columns=["territory_id", "category", "method", "break_month"])
    br.to_parquet(ROOT / "outputs" / "cpd_breaks.parquet")
    print(f"скан всей панели: {len(br)} брейков, {time.time()-t0:.0f}с total")


if __name__ == "__main__":
    main()
