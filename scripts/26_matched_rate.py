"""Matched detection rate сравнение CPD (К3, R04 §2.3.3б — перенос приёма на слой 1).

Идея: всем детекторам разрешено ОДИНАКОВОЕ число брейков на панели — K = число брейков
эталонного PELT-BIC (pen = 2·log(24) на стандартизованном ряду z/level_scale(z)) на той же
измерительной панели с той же инъекцией. Каждый метод отдаёт свои топ-K кандидатов по
собственному скору; recall/F1@±2 при равной частоте срабатываний изолирует качество
ЛОКАЛИЗАЦИИ от агрессивности порога (в v2 сравнение на замороженных FAR-точках смешивало
и то, и другое: Window pen=4 почти молчит, CUSUM h=20 стреляет в 8 раз чаще PELT).

Скоры по позициям (единый вход — десезонированный лог-ряд, как в 06):
- PELT/BinSeg: плотный прогон (pen=1.0 на стандартизованном ряду) → скор брейка =
  прирост разбиения (cost-gain) в единицах σ²: sse(prev,next) − sse(prev,b) − sse(b,next);
- Window: внутренняя rbf-discrepancy кривая ruptures (jump=1), позиции width..n−width;
- CUSUM: двухсторонний кумулянт без рестарта, скор = max(sp,|sn|)/σ_уровня;
- EWMA: |z_t − μ0|/(σ√(λ/(2−λ))), μ0 = медиана первых 6, λ=0.25;
- BOCPD: p_restart из src.cpd.bocpd_prestart (уже вероятность — нативный скор);
- PELT-BIC: эталон, задаёт K (его строка = якорь: top-K по его gain это он сам).

Отбор топ-K: сортировка всех кандидатов панели по скору, жадный набор с non-maximum
suppression ±2 мес внутри ряда (иначе резкий пик кривой тратит весь бюджет на соседние
позиции одного ряда — дубли в окне ±2 по eval_detection и так не штрафуются, но бюджет
при этом сгорает впустую; NMS — дизайн-решение этого скрипта, зафиксировано здесь).

План инъекций и измерительная подвыборка — идентичны 06 (seed 42/123, src.cpd.CELLS) →
сравнение с v2-таблицами парное. Выход: outputs/cpd_matched_rate_v3.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import (category_profiles, deseasonalize, mad_scale, level_scale,  # noqa: E402
                     bocpd_prestart, inject, eval_detection, load_wide,
                     stratified_measurement_idx, injection_plans,
                     CELLS, K_SHRINK, DETREND, ROBUST)

PEN_BIC = 2.0 * np.log(24)      # BIC для l2 на стандартизованном ряду (p=2: mean+break)
PEN_DENSE = 1.0                 # плотный прогон PELT/BinSeg для кандидатов (σ²-шкала)
POS_LO, POS_HI = 3, 21          # кандидаты 3..20 (min_size=3 симметрично, как у ruptures)
NMS_TOL = 2                     # non-maximum suppression внутри ряда


# --------------------------------------------------------------------------- #
# Скоры кандидатов (единый интерфейс: z → dict позиция → скор, позиции 1..23)
# --------------------------------------------------------------------------- #

def _seg_gains(zs: np.ndarray, breaks: list[int]) -> dict[int, float]:
    """Cost-gain каждого брейка разбиения в σ²-единицах (l2/SSE по стандартизованному)."""
    n = len(zs)
    c1 = np.concatenate([[0.0], np.cumsum(zs)])
    c2 = np.concatenate([[0.0], np.cumsum(zs * zs)])

    def sse(a, b):
        m = b - a
        s1, s2 = c1[b] - c1[a], c2[b] - c2[a]
        return s2 - s1 * s1 / m

    bounds = [0] + sorted(breaks) + [n]
    gains = {}
    for k in range(1, len(bounds) - 1):
        a, b, c = bounds[k - 1], bounds[k], bounds[k + 1]
        gains[b] = sse(a, c) - sse(a, b) - sse(b, c)
    return gains


def scores_pelt(z: np.ndarray) -> dict[int, float]:
    zs = z / level_scale(z)
    bk = [b for b in rpt.Pelt(model="l2", min_size=3).fit(zs).predict(pen=PEN_DENSE) if b < len(zs)]
    return _seg_gains(zs, bk)


def scores_binseg(z: np.ndarray) -> dict[int, float]:
    zs = z / level_scale(z)
    bk = [b for b in rpt.Binseg(model="l2", min_size=3).fit(zs).predict(pen=PEN_DENSE) if b < len(zs)]
    return _seg_gains(zs, bk)


def scores_window(z: np.ndarray) -> dict[int, float]:
    det = rpt.Window(width=4, model="rbf", min_size=3, jump=1).fit(z)
    # в ruptures 1.1.10 score выровнен по det.inds (полуокно width//2), НЕ по width..n−width
    return {int(k): float(v) for k, v in zip(det.inds, det.score)}


def scores_cusum(z: np.ndarray) -> dict[int, float]:
    s = level_scale(z)
    k = 0.5 * s
    mu = float(np.median(z[:6]))
    sp, sn = 0.0, 0.0
    out = {}
    for t in range(1, len(z)):
        e = z[t] - mu
        sp, sn = max(0.0, sp + e - k), min(0.0, sn + e + k)
        out[t] = max(sp, abs(sn)) / s
    return out


def scores_ewma(z: np.ndarray, lam: float = 0.25) -> dict[int, float]:
    s = level_scale(z)
    mu = float(np.median(z[:6]))
    norm = s * np.sqrt(lam / (2 - lam))
    zt = mu
    out = {}
    for t in range(1, len(z)):
        zt = lam * z[t] + (1 - lam) * zt
        out[t] = abs(zt - mu) / norm
    return out


def scores_bocpd(z: np.ndarray) -> dict[int, float]:
    p = bocpd_prestart(z)
    return {t: float(p[t]) for t in range(1, len(z))}


SCORERS = {"PELT": scores_pelt, "BinSeg": scores_binseg, "Window": scores_window,
           "CUSUM": scores_cusum, "EWMA": scores_ewma, "BOCPD": scores_bocpd}


def breaks_pelt_bic(z: np.ndarray) -> list[int]:
    """Эталон числа брейков: PELT-l2, pen = 2·log(24) на ряде, стандартизованном level_scale."""
    zs = z / level_scale(z)
    return [b for b in rpt.Pelt(model="l2", min_size=3).fit(zs).predict(pen=PEN_BIC) if b < len(zs)]


def select_top_k(cands: list[tuple[float, int, int]], k: int,
                 nms_tol: int = NMS_TOL) -> dict[int, list[int]]:
    """Топ-K панели: cands = (score, series_j, pos) → {series_j: [pos,...]} с NMS внутри ряда."""
    cands = sorted(cands, key=lambda c: -c[0])
    picked: dict[int, list[int]] = {}
    n = 0
    for sc, j, pos in cands:
        if n >= k:
            break
        got = picked.setdefault(j, [])
        if any(abs(pos - p) <= nms_tol for p in got):
            continue
        got.append(pos)
        n += 1
    return picked


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)

    meas = stratified_measurement_idx(W, cats)
    plans = injection_plans(len(meas))
    print(f"измерительных рядов: {len(meas)}; ячеек: {len(CELLS)}", flush=True)

    rows = []
    for kind, d in CELLS:
        taus = plans[(kind, d)]
        # --- эталонный бюджет: суммарное число брейков PELT-BIC на этой ячейке
        zs_cell, k_cell = [], 0
        for j, i in enumerate(meas):
            y = W[i]
            z = ds(inject(y, kind, int(taus[j]), d, s=mad_scale(y)), i)
            zs_cell.append(z)
            k_cell += len(breaks_pelt_bic(z))
        # --- кандидаты каждого метода
        per_method = {m: [] for m in SCORERS}
        for j, z in enumerate(zs_cell):
            for m, fn in SCORERS.items():
                for pos, sc in fn(z).items():
                    if POS_LO <= pos < POS_HI and np.isfinite(sc):
                        per_method[m].append((sc, j, pos))
        # --- отбор топ-K и оценка
        for m in SCORERS:
            picked = select_top_k(per_method[m], k_cell)
            tp = fp = fn = 0
            delays = []
            for j in range(len(meas)):
                r = eval_detection(picked.get(j, []), int(taus[j]), tol=2)
                tp += r[0]; fp += r[1]; fn += r[2]
                if not np.isnan(r[3]):
                    delays.append(r[3])
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            rows.append(dict(method=m, kind=kind, delta=d, K_panel=k_cell,
                             n_selected=sum(len(v) for v in picked.values()),
                             tp=tp, fp=fp, fn=fn, precision=prec, recall=rec,
                             F1=2 * prec * rec / max(prec + rec, 1e-9),
                             delay_mean=float(np.mean(delays)) if delays else np.nan,
                             n_series=len(meas)))
        print(f"ячейка {kind} {d}: K={k_cell}, {time.time()-t0:.0f}с", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "cpd_matched_rate_v3.csv", index=False)
    print(out[out.kind == "level"].round(3).to_string(index=False), flush=True)
    print(f"total {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
