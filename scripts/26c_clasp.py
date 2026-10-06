"""ClaSP (sktime.detection, M4 ростера R04) на 200-рядовой подвыборке инъекций.

Установка: sktime 1.2.0 поставлена через `uv pip install` (+scikit-base 1.1.1, numpy НЕ
тронут — лог outputs/logs/26c_sktime_install.log); в sktime 1.x модуль переименован
sktime.annotation → sktime.detection (C1 ссылался на annotation).

Прогон: 200 рядов из измерительной подвыборки 06 (те же ряды/инъекции, seed 42/123 —
парность с PELT/Window), все 10 ячеек CELLS. Вход — тот же десезонированный лог-ряд
(единый вход R04 §1.3). period_length=6: при n=24 дефолтный 10 оставляет профилю
всего 15 позиций и съедает края; 6 = компромисс «локальность vs статистика окна»
(n_cps=1: на 24 точках больше одного честно различимого брейка нет, C1 §0).

Сравнение на тех же рядах/инъекциях: PELT и Window на замороженных рабочих точках
(«как в проде», cpd_calibration_v2.csv) + контрольная ячейка без инъекции (FPR:
доля рядов с ≥1 брейком) — ClaSP в этом протоколе без отдельной калибровки (parameter-
light и есть его аргумент), поэтому сравнение по F1 — при РАЗНЫХ operating points;
FPR-строка показывает цену этого.

Выход: outputs/cpd_clasp_v3.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import (category_profiles, deseasonalize, mad_scale, inject,  # noqa: E402
                     eval_detection, load_wide, stratified_measurement_idx,
                     injection_plans, det_pelt, det_window,
                     CELLS, K_SHRINK, DETREND, ROBUST)

from sktime.detection.clasp import ClaSPSegmentation  # noqa: E402

N_SUB = 200
PERIOD = 6


def det_clasp(z: np.ndarray) -> list[int]:
    r = ClaSPSegmentation(period_length=PERIOD, n_cps=1).fit_predict(np.asarray(z, float))
    # sktime 1.2: fit_predict возвращает DataFrame с колонкой "ilocs"
    locs = r["ilocs"].to_numpy() if hasattr(r, "columns") else np.atleast_1d(r)
    return [int(b) for b in np.atleast_1d(locs) if 0 < b < len(z)]


def f1_row(method, recs, n):
    arr = np.array([(r[0], r[1], r[2]) for r in recs], float)
    tp, fp, fn = arr.sum(axis=0)
    delays = np.array([r[3] for r in recs], float)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return dict(method=method, tp=int(tp), fp=int(fp), fn=int(fn),
                precision=prec, recall=rec, F1=2 * prec * rec / max(prec + rec, 1e-9),
                delay_mean=float(np.nanmean(delays)) if np.isfinite(delays).any() else np.nan,
                n_series=n)


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)

    cal = pd.read_csv(ROOT / "outputs" / "cpd_calibration_v2.csv")
    frozen = {r.method: float(r.param) for r in cal[cal.is_working].itertuples()}

    meas = stratified_measurement_idx(W, cats)[:N_SUB]
    plans = injection_plans(len(stratified_measurement_idx(W, cats)))
    print(f"подвыборка: {len(meas)} рядов; ClaSP period={PERIOD}", flush=True)

    # контроль (без инъекции): доля рядов с ≥1 брейком
    rows = []
    for m, fn in (("ClaSP", det_clasp), ("PELT", lambda z: det_pelt(z, frozen["PELT"])),
                  ("Window", lambda z: det_window(z, frozen["Window"]))):
        hits = sum(1 for j, i in enumerate(meas) if len(fn(ds(W[i], i))) > 0)
        rows.append(dict(method=m, kind="control", delta=np.nan, fpr=hits / len(meas),
                         n_series=len(meas)))
        print(f"контроль {m}: FPR={hits/len(meas):.3f}, {time.time()-t0:.0f}с", flush=True)

    for kind, d in CELLS:
        taus = plans[(kind, d)]
        recs = {m: [] for m in ("ClaSP", "PELT", "Window")}
        for j, i in enumerate(meas):
            y = W[i]
            z = ds(inject(y, kind, int(taus[j]), d, s=mad_scale(y)), i)
            recs["ClaSP"].append(eval_detection(det_clasp(z), int(taus[j]), tol=2))
            recs["PELT"].append(eval_detection(det_pelt(z, frozen["PELT"]), int(taus[j]), tol=2))
            recs["Window"].append(eval_detection(det_window(z, frozen["Window"]), int(taus[j]), tol=2))
        for m in recs:
            rows.append(dict(kind=kind, delta=d, **f1_row(m, recs[m], len(meas))))
        print(f"ячейка {kind} {d}: {time.time()-t0:.0f}с", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "cpd_clasp_v3.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)
    print(f"total {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
