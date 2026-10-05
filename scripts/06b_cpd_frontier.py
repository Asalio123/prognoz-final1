"""CPD фронтир F1×FAR (К3, R04): честное сравнение детекторов кривыми, не одной точкой.

Сетка штрафов на детектор × дельты инъекций {0.4, 0.8, 1.6, 3.0}σ (level) и {0.02, 0.04} лог/мес
(trend) на 200 стратифицированных рядах. FAR считается на чистых рядах той же подвыборки.
Дисплей-точка: FAR ≤ 1.2/1000 МО-мес (бюджет радара). Выход: outputs/cpd_frontier.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def deseasonalize(y):
    y = np.asarray(y, float)
    prof = np.array([np.median(y[m::12]) for m in range(12)])
    return y - prof[np.arange(len(y)) % 12]


def det_pelt(y, pen):
    return [b for b in rpt.Pelt(model="l2").fit(y).predict(pen=pen) if b < len(y)]


def det_binseg(y, pen):
    return [b for b in rpt.Binseg(model="l2").fit(y).predict(pen=pen) if b < len(y)]


def det_window(y, pen):
    return [b for b in rpt.Window(width=4, model="rbf").fit(y).predict(pen=pen) if b < len(y)]


def det_cusum(y, h_sigma):
    s = np.std(np.diff(y)) or 1.0
    k, thr = 0.5 * s, h_sigma * s
    breaks, sp, sn = [], 0.0, 0.0
    mu = np.median(y[:6])
    for t in range(1, len(y)):
        e = y[t] - mu
        sp, sn = max(0, sp + e - k), min(0, sn + e + k)
        if sp > thr or abs(sn) > thr:
            breaks.append(t)
            sp, sn = 0.0, 0.0
            mu = np.median(y[max(0, t - 5): t + 1])
    return breaks


def det_bocpd(y, q):
    from scipy.special import logsumexp
    from scipy.stats import norm
    n = len(y)
    mu0, kappa = np.median(y[:6]), 1.0
    s2_0 = (np.std(np.diff(y)) or 1.0) ** 2
    H = 1 / 12
    logR = np.full((n + 1, n + 1), -np.inf)
    logR[0, 0] = 0.0
    for t in range(1, n + 1):
        lp = []
        for r in range(t):
            d = y[t - r: t]
            m = len(d)
            mu_p = (mu0 * kappa + d.sum()) / (kappa + m)
            s2 = s2_0 * (1 + 1 / (kappa + m))
            lp.append(norm.logpdf(y[t - 1], loc=mu_p, scale=np.sqrt(s2)))
        lp = np.array(lp)
        growth = logR[t - 1, :t] + lp + np.log(1 - H)
        cp = logR[t - 1, :t] + lp + np.log(H)
        logR[t, :t] = growth
        logR[t, t] = logsumexp(cp)
        logR[t] -= logsumexp(logR[t])
    p_restart = np.exp([logR[t, t] for t in range(n + 1)])
    thr = max(np.quantile(p_restart[1:], q), 0.05)
    return [t for t in range(1, n) if p_restart[t] > thr]


def inject(y, kind, tau, delta):
    s = np.std(np.diff(y)) or 1.0
    z = y.copy()
    if kind == "level":
        z[tau:] += delta * s
    elif kind == "trend":
        z[tau:] += delta * s * np.arange(1, len(y) - tau + 1)
    return z


GRID = {
    "PELT": ("pen", det_pelt, [0.5, 1.0, 2.0, 4.0, 8.0]),
    "BinSeg": ("pen", det_binseg, [0.5, 1.0, 2.0, 4.0, 8.0]),
    "Window": ("pen", det_window, [1.0, 3.0, 6.0, 12.0, 24.0]),
    "CUSUM": ("h", det_cusum, [4.0, 6.0, 8.0, 12.0]),
    "BOCPD": ("q", det_bocpd, [0.90, 0.95, 0.99, 0.995]),
}
DELTAS_LEVEL = [0.4, 0.8, 1.6, 3.0]
DELTAS_TREND = [0.04, 0.08]


def main():
    t0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    rng = np.random.default_rng(42)
    sub = wide.iloc[rng.choice(len(wide), size=200, replace=False)].to_numpy()

    rows = []
    for name, (pname, fn, grid) in GRID.items():
        for g in grid:
            # FAR на чистых
            nb = sum(len(fn(deseasonalize(y), g)) if pname != "q" else len(fn(deseasonalize(y), g))
                     for y in sub)
            far = nb / (len(sub) * 24)
            for kind, deltas in [("level", DELTAS_LEVEL), ("trend", DELTAS_TREND)]:
                for delta in deltas:
                    tps, fps, fns, delays = 0, 0, 0, []
                    for y in sub:
                        tau = int(rng.integers(8, 17))
                        z = deseasonalize(inject(y, kind, tau, delta))
                        found = fn(z, g)
                        hits = [f for f in found if abs(f - tau) <= 2]
                        tps += 1 if hits else 0
                        fps += len([f for f in found if abs(f - tau) > 2])
                        fns += 0 if hits else 1
                        if hits:
                            delays.append(min(hits) - tau)
                    prec = tps / max(tps + fps, 1)
                    rec = tps / max(tps + fns, 1)
                    rows.append(dict(method=name, param=g, kind=kind, delta=delta,
                                     far_per_1000=far * 1000, precision=prec, recall=rec,
                                     f1=2 * prec * rec / max(prec + rec, 1e-9),
                                     delay=np.mean(delays) if delays else np.nan))
            print(f"{name} param={g} done, {time.time()-t0:.0f}с", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "cpd_frontier.csv", index=False)
    # дисплей: лучший F1 при FAR <= 1.2/1000 по каждому методу и типу
    ok = out[out.far_per_1000 <= 1.2]
    best = ok.loc[ok.groupby(["method", "kind", "delta"])["f1"].idxmax()] if len(ok) else out.head(0)
    print(best.round(3).to_string(index=False))
    print(f"total {time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
