"""CPD-зоопарк v2 (К3): disjoint калибровка → замороженные пороги → парные инъекции.

Фиксы аудита A11/A05:
- калибровка отделена от измерения: FAR-таблицы строятся на всей панели МИНУС
  измерительные 400 рядов (в rc0 пороги подбирались на тех же рядах, что мерялась
  мощность); рабочие точки замораживаются ДО измерения recall (src.cpd.working_points);
- дизайн парный: один план инъекций (ряд × tau × тип × δ, seed 123) для всех методов;
  добавлены отрицательные δ (двусторонность, A05 P1-7) и variance-инъекции;
- десезонирование из src.cpd (ridge+Huber со шринкажем к профилю категории) вместо
  median-of-2, зеркалившего брейки; Window с jump=1 (дефолтный jump=5 квантовал
  брейки в {5,10,15,20} — причина «idx15-артефакта» rc0);
- BOCPD — корректная рекурсия Adams–MacKay, порог абсолютный и калиброванный;
  CUSUM — шкала по MAD приростов; добавлен EWMA (R04 M8).

Выходы: outputs/cpd_calibration_v2.csv (порог→FAR, рабочие точки),
outputs/cpd_injection_scores_v2.csv (мощность на 400 парных рядах),
outputs/cpd_breaks_v2.parquet (скан всей панели замороженными детекторами).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import (category_profiles, deseasonalize, mad_scale, bocpd_prestart,  # noqa: E402
                     inject, eval_detection, load_wide, stratified_measurement_idx,
                     injection_plans, detect_all, working_points,
                     GRIDS, CELLS, FAR_BUDGET, K_SHRINK, DETREND, ROBUST)


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)
    print(f"рядов: {len(W)}; конфиг: k={K_SHRINK}, detrend={DETREND}, huber={ROBUST}", flush=True)

    # --- сплит: измерительные 400 (стратифицированные) vs калибровочная панель
    meas = stratified_measurement_idx(W, cats)
    is_meas = np.zeros(len(W), bool)
    is_meas[meas] = True
    cal_idx = np.where(~is_meas)[0]
    print(f"калибровка: {len(cal_idx)} рядов; измерение: {len(meas)}", flush=True)

    # --- десезонированные калибровочные ряды + BOCPD p_restart (один прогон на ряд)
    Z_cal = np.vstack([ds(W[i], i) for i in cal_idx])
    print(f"десезонирование калибровки: {time.time()-t0:.0f}с", flush=True)
    P_cal = [bocpd_prestart(z) for z in Z_cal]
    print(f"BOCPD калибровка: {time.time()-t0:.0f}с", flush=True)

    # --- 1. калибровочная карта: порог → FAR на чистой панели
    cal_rows = []
    for name, (pname, fn, grid) in GRIDS.items():
        for g in grid:
            if name == "BOCPD":
                nb = sum(int((p > g).sum()) for p in P_cal)
            else:
                nb = sum(len(fn(z, g)) for z in Z_cal)
            cal_rows.append(dict(method=name, param=g, n_breaks=nb,
                                 far_per_1000=nb / (len(cal_idx) * 24) * 1000,
                                 n_series_cal=len(cal_idx)))
        print(f"калибровка {name}: {time.time()-t0:.0f}с", flush=True)
    cal = pd.DataFrame(cal_rows)
    frozen = working_points(cal, FAR_BUDGET)
    cal["is_working"] = [(r.method in frozen and frozen[r.method] is not None
                          and abs(r.param - frozen[r.method]) < 1e-12)
                         for r in cal.itertuples()]
    cal.to_csv(ROOT / "outputs" / "cpd_calibration_v2.csv", index=False)
    print(f"замороженные точки (бюджет {FAR_BUDGET}/1000): {frozen}", flush=True)
    print(cal[cal.is_working].round(3).to_string(index=False), flush=True)

    # --- 2. парный план инъекций + измерение мощности на замороженных порогах
    plans = injection_plans(len(meas))
    rows = []
    for kind, d in CELLS:
        taus = plans[(kind, d)]
        per_series = {m: [] for m in GRIDS if frozen[m] is not None}
        for j, i in enumerate(meas):
            y = W[i]
            s = mad_scale(y)
            z = ds(inject(y, kind, int(taus[j]), d, s=s), i)
            p_restart = bocpd_prestart(z)
            for m in per_series:
                found = detect_all(z, p_restart, m, frozen[m])
                per_series[m].append(eval_detection(found, int(taus[j])))
        for m, recs in per_series.items():
            arr = np.array([(r[0], r[1], r[2]) for r in recs], float)
            tp, fp, fn_ = arr.sum(axis=0)
            delays = np.array([r[3] for r in recs], float)
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn_, 1)
            rows.append(dict(method=m, kind=kind, delta=d, param=frozen[m],
                             far_per_1000=cal[(cal.method == m) & cal.is_working].far_per_1000.iloc[0],
                             precision=prec, recall=rec,
                             F1=2 * prec * rec / max(prec + rec, 1e-9),
                             delay=np.nanmean(delays), n=len(meas)))
        print(f"инъекция {kind} {d}: {time.time()-t0:.0f}с", flush=True)
    inj = pd.DataFrame(rows)
    inj.to_csv(ROOT / "outputs" / "cpd_injection_scores_v2.csv", index=False)
    print(inj[inj.kind == "level"].round(3).to_string(index=False), flush=True)

    # --- 3. полный скан панели замороженными детекторами (для кейсов и радара)
    full = []
    for i, sid in enumerate(wide.index):
        z = ds(W[i], i)
        p_restart = bocpd_prestart(z)
        for m in GRIDS:
            if frozen[m] is None:
                continue
            for b in detect_all(z, p_restart, m, frozen[m]):
                full.append((sid.split("|")[0], sid.split("|")[1], m, b))
        if (i + 1) % 3000 == 0:
            print(f"скан {i+1}/{len(W)}, {time.time()-t0:.0f}с", flush=True)
    br = pd.DataFrame(full, columns=["territory_id", "category", "method", "break_month"])
    br.to_parquet(ROOT / "outputs" / "cpd_breaks_v2.parquet")
    print(f"скан: {len(br)} брейков; total {time.time()-t0:.0f}с", flush=True)
    print(br.groupby("method").size().to_string(), flush=True)


if __name__ == "__main__":
    main()
