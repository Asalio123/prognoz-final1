"""NAB-скоринг CPD-детекторов (К3, R04 §2.2 / C1 §4.3): единственная схема, где РАННЯЯ
детекция формально вознаграждается — прямое попадание в формулировку трека.

Канонический Numenta Anomaly Benchmark (Lavin & Ahmad 2015):
- окно аномалии [tau−2, tau+2] мес (по брифу, ±2 — согласовано с окном F1);
- детекция в окне: score = w_tp · σ(p), p = 2(t−w0)/(w1−w0) − 1 ∈ [−1,1],
  σ(p) = sign(p)·(2/(1+e^{5|p|}) − 1) — scaled sigmoid NAB: +0.987 на ЛЕВОМ краю окна
  (за 2 мес до шока — «заранее выявлен»), 0 ровно в tau, −0.987 на правом краю
  (опоздавшая детекция штрафуется почти как промах). Семантика окна зафиксирована здесь:
  ранняя сторона окна = предиктивная зона, поздняя = догоняющая;
- детекций в окне несколько → засчитывается первая (максимальный score), остальные в том
  же окне игнорируются (не FP), как в NAB;
- каждая детекция вне окна = −w_fp; окно без детекции = −w_fn;
- нормализация корпуса: (S − S_null)/(S_perfect − S_null), где S_null = n·(−w_fn)
  (молчащий детектор), S_perfect = n·w_tp·σ(−1) (все ряды детектированы на левом краю).

Профили: standard (w_fp=0.11) и reward low FP (w_fp=0.22, профиль панели 2016 рядов —
дешёвый FP-rate недопустим); w_tp=w_fn=1 в обоих (веса NAB-дефолта).

Два режима (оба парные с 06/26: те же 378 рядов, тот же план инъекций seed 123):
- frozen: детекторы на замороженных рабочих точках cpd_calibration_v2.csv («как в проде»,
  бюджет FAR 2/1000) — мощность мала, NAB в основном дифференцирует по FP-штрафам;
- matched: топ-K по скору каждого метода (K = число брейков PELT-BIC на ячейке, скоры из
  26_matched_rate) — дифференцирует качество локализации/ранности при равной частоте.

Выход: outputs/cpd_nab_v3.csv (строки method × regime × profile × ячейка + агрегат ALL).
"""
import importlib
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
mr = importlib.import_module("26_matched_rate")  # скоры/селекция — единый источник

from src.cpd import (category_profiles, deseasonalize, mad_scale, bocpd_prestart,  # noqa: E402
                     inject, load_wide, stratified_measurement_idx, injection_plans,
                     detect_all, CELLS, K_SHRINK, DETREND, ROBUST)

WINDOW = 2
PROFILES = {"standard": dict(tp=1.0, fp=0.11, fn=1.0),
            "reward_low_fp": dict(tp=1.0, fp=0.22, fn=1.0)}


def scaled_sigmoid(p: float) -> float:
    s = 2.0 / (1.0 + math.exp(5 * abs(p))) - 1.0
    return s if p >= 0 else -s


def nab_series(detections: list[int], tau: int, w: dict, window: int = WINDOW):
    """(score, tp, fp, fn) одного ряда: первое попадание в окно + FP вне окон."""
    w0, w1 = tau - window, tau + window
    dets = sorted(detections)
    inside = [t for t in dets if w0 <= t <= w1]
    if inside:
        p = 2 * (inside[0] - w0) / (w1 - w0) - 1.0
        score = w["tp"] * scaled_sigmoid(p)
        tp, fn = 1, 0
        fp = len([t for t in dets if not (w0 <= t <= w1)])
    else:
        score = -w["fn"]
        tp, fn = 0, 1
        fp = len(dets)
    return score - w["fp"] * fp, tp, fp, fn


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)

    cal = pd.read_csv(ROOT / "outputs" / "cpd_calibration_v2.csv")
    frozen = {r.method: float(r.param) for r in cal[cal.is_working].itertuples()}
    print(f"замороженные точки: {frozen}", flush=True)

    meas = stratified_measurement_idx(W, cats)
    plans = injection_plans(len(meas))
    methods = list(mr.SCORERS)

    rows = []
    for kind, d in CELLS:
        taus = plans[(kind, d)]
        det_frozen = {m: [] for m in methods}     # per-series списки брейков (frozen)
        cands = {m: [] for m in methods}          # (score, j, pos) для matched
        k_cell = 0
        for j, i in enumerate(meas):
            y = W[i]
            z = ds(inject(y, kind, int(taus[j]), d, s=mad_scale(y)), i)
            p_restart = bocpd_prestart(z)
            k_cell += len(mr.breaks_pelt_bic(z))
            for m in methods:
                det_frozen[m].append(detect_all(z, p_restart, m, frozen[m]))
                for pos, sc in mr.SCORERS[m](z).items():
                    if mr.POS_LO <= pos < mr.POS_HI and np.isfinite(sc):
                        cands[m].append((sc, j, pos))
        det_matched = {m: mr.select_top_k(cands[m], k_cell) for m in methods}

        for regime, det in (("frozen", det_frozen), ("matched", det_matched)):
            for m in methods:
                for prof, w in PROFILES.items():
                    tot = tp = fp = fn = 0
                    sc_sum = 0.0
                    for j in range(len(meas)):
                        found = det[m][j] if regime == "frozen" else det[m].get(j, [])
                        s, a, b, c = nab_series(found, int(taus[j]), w)
                        sc_sum += s; tp += a; fp += b; fn += c
                    n = len(meas)
                    s_null = -n * w["fn"]
                    s_perf = n * w["tp"] * scaled_sigmoid(-1.0)
                    rows.append(dict(regime=regime, profile=prof, method=m, kind=kind,
                                     delta=d, n_series=n, K_panel=(k_cell if regime == "matched" else np.nan),
                                     tp=tp, fp=fp, fn=fn, nab_raw=sc_sum,
                                     nab_null=s_null, nab_perfect=s_perf,
                                     nab_norm=(sc_sum - s_null) / (s_perf - s_null)))
        print(f"ячейка {kind} {d}: {time.time()-t0:.0f}с", flush=True)

    out = pd.DataFrame(rows)
    agg = []
    for (regime, prof, m), g in out.groupby(["regime", "profile", "method"]):
        s_null, s_perf = g.nab_null.sum(), g.nab_perfect.sum()
        agg.append(dict(regime=regime, profile=prof, method=m, kind="ALL", delta=np.nan,
                        n_series=int(g.n_series.sum()), K_panel=np.nan,
                        tp=int(g.tp.sum()), fp=int(g.fp.sum()), fn=int(g.fn.sum()),
                        nab_raw=g.nab_raw.sum(), nab_null=s_null, nab_perfect=s_perf,
                        nab_norm=(g.nab_raw.sum() - s_null) / (s_perf - s_null)))
    out = pd.concat([out, pd.DataFrame(agg)], ignore_index=True)
    out.to_csv(ROOT / "outputs" / "cpd_nab_v3.csv", index=False)
    show = out[(out.kind == "ALL")].sort_values(["regime", "profile", "nab_norm"],
                                                ascending=[True, True, False])
    print(show.round(3).to_string(index=False), flush=True)
    print(f"total {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
