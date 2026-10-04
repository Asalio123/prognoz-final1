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


def det_bocpd(y, hazard=1/12, q=0.995):
    """Hand-rolled Gaussian BOCPD (Adams–MacKay): пакет с scipy.misc.comb сломан на scipy 1.16,
    онлайн-StudentT из пакета не калибруется на 24 точках. Матрица run-length вероятностей,
    брейк = P(run=0) > q-квантиль фона."""
    n = len(y)
    mu0, kappa, s2_0 = np.median(y[:6]), 1.0, (np.std(np.diff(y)) or 1.0) ** 2
    logR = np.full((n + 1, n + 1), -np.inf)
    logR[0, 0] = 0.0
    H = hazard
    # параметры predictive (Normal-Inverse-Gamma упрощение: Student-t с фикс. дисперсией)
    def logpred(t, r):
        # данные рана: y[t-r:t]
        d = y[t - r: t]
        m = len(d)
        mu_p = (mu0 * kappa + d.sum()) / (kappa + m)
        s2 = s2_0 * (1 + 1 / (kappa + m))
        from scipy.stats import norm
        return norm.logpdf(y[t], loc=mu_p, scale=np.sqrt(s2))
    for t in range(1, n + 1):
        growth = np.array([logR[t - 1, r] + logpred(t - 1, r) + np.log(1 - H) for r in range(t)])
        cp = np.array([logR[t - 1, r] + logpred(t - 1, r) + np.log(H) for r in range(t)])
        logR[t, :t] = growth
        from scipy.special import logsumexp
        logR[t, t] = logsumexp(cp)  # сброс рана = брейк перед t
        logR[t] -= logsumexp(logR[t])
    p_restart = np.exp([logR[t, t] for t in range(n + 1)])
    bg = np.quantile(p_restart[1:], q)
    return [t for t in range(1, n) if p_restart[t] > max(bg, 0.3)]


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
    # калибровка под FAR-бюджет (R04: пороги диктуются бюджетом ДО измерения delay)
    sub_mat = sub.to_numpy()
    target = 1.2 / 1000
    pens = {}
    grid = {"PELT": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0], "BinSeg": [0.5, 1.0, 2.0, 4.0, 8.0, 16.0],
            "Window": [1.0, 3.0, 6.0, 12.0, 24.0, 48.0]}
    for name, fn in [("PELT", det_pelt), ("BinSeg", det_binseg), ("Window", det_window)]:
        for pen in grid[name]:
            nb = sum(len(fn(deseasonalize(row), pen)) for row in sub_mat)
            far = nb / (len(sub) * 24)
            if far <= target:
                pens[name] = pen
                break
        else:
            pens[name] = grid[name][-1]
        far_f = sum(len(fn(deseasonalize(row), pens[name])) for row in sub_mat) / (len(sub) * 24)
        print(f"{name}: pen={pens[name]}, FAR={far_f*1000:.2f}/1000 (бюджет 1.2)", flush=True)
    for hs in [4.0, 6.0, 8.0, 12.0]:
        nb = sum(len(det_cusum(deseasonalize(row), h_sigma=hs)) for row in sub_mat[:200])
        if nb / (200 * 24) <= target:
            break
    CUSUM_H = hs
    print(f"CUSUM: h_sigma={CUSUM_H}, FAR={nb/(200*24)*1000:.2f}/1000", flush=True)
    nb = sum(len(det_bocpd(deseasonalize(row))) for row in sub_mat[:100])
    print(f"BOCPD: FAR={nb/(100*24)*1000:.2f}/1000", flush=True)

    # --- инъекционный стенд
    rows_out = []
    for kind in ["level", "trend", "variance"]:
        for delta in ([0.2, 0.4, 0.8] if kind != "variance" else [1.5, 2.0]):
            for i, row in enumerate(sub.to_numpy()[:200]):
                tau = int(rng.integers(8, 17))
                z = inject(row, kind, tau, delta)
                z_ds = deseasonalize(z)
                for name, fn in METHODS.items():
                    if name in ("PELT", "BinSeg", "Window"):
                        found = fn(z_ds, pens[name])
                    elif name == "CUSUM":
                        found = fn(z_ds, h_sigma=CUSUM_H)
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
            if name in ("PELT", "BinSeg", "Window"):
                bk = fn(y_ds, pens[name])
            elif name == "CUSUM":
                bk = fn(y_ds, h_sigma=CUSUM_H)
            else:
                bk = fn(y_ds)
            for b in bk:
                full.append((tid, cat, name, b))
    br = pd.DataFrame(full, columns=["territory_id", "category", "method", "break_month"])
    br.to_parquet(ROOT / "outputs" / "cpd_breaks.parquet")
    print(f"скан всей панели: {len(br)} брейков, {time.time()-t0:.0f}с total")


if __name__ == "__main__":
    main()
