"""CPD фронтир v2 (К3, R04): FAR на полной калибровочной панели × парные инъекции × CI.

Фиксы аудита A11/A05:
- FAR каждой точки сетки измерен на всей панели минус измерительные 400 (в rc0 — на
  тех же 200 рядах: гранулярность 1 ложный брейк = 0.21/1000, и калибровка сливалась
  с измерением);
- парный дизайн: план инъекций фиксирован (seed 123) — все методы и точки сетки видят
  ОДНИ и те же инъекции (в rc0 tau перегенерировался в каждой ячейке сетки);
- cluster-bootstrap по ряду (2000 ресэмплов) → CI для F1 каждой ячейки;
- ростер и сетки унифицированы с 06_run_cpd.py через src.cpd (конец двум разным
  «BOCPD» под одним именем); добавлены отрицательные δ и variance-инъекции.

Рабочие точки выбираются из FAR-таблицы ДО измерения recall (src.cpd.working_points,
бюджет 2/1000); если у метода нет точки между 10/1000 и 0 — это фиксируется как
находка, а не маскируется. Выход: outputs/cpd_frontier_v2.csv.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import (category_profiles, deseasonalize, mad_scale, bocpd_prestart,  # noqa: E402
                     inject, load_wide, stratified_measurement_idx, injection_plans,
                     detect_all, working_points, GRIDS, CELLS, FAR_BUDGET,
                     K_SHRINK, DETREND, ROBUST)

N_BOOT = 2000
SEED_BOOT = 7


def f1_of(arr):
    """arr: (n, 3) tp/fp/fn по рядам → (P, R, F1)."""
    tp, fp, fn_ = arr.sum(axis=0)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def main():
    t0 = time.time()
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    ds = lambda y, i: deseasonalize(y, prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)

    meas = stratified_measurement_idx(W, cats)
    is_meas = np.zeros(len(W), bool)
    is_meas[meas] = True
    cal_idx = np.where(~is_meas)[0]
    print(f"калибровка {len(cal_idx)} / измерение {len(meas)} рядов", flush=True)

    # --- FAR-таблица на полной калибровочной панели
    Z_cal = np.vstack([ds(W[i], i) for i in cal_idx])
    P_cal = [bocpd_prestart(z) for z in Z_cal]
    cal_rows = []
    for name, (_, fn, grid) in GRIDS.items():
        for g in grid:
            nb = (sum(int((p > g).sum()) for p in P_cal) if name == "BOCPD"
                  else sum(len(fn(z, g)) for z in Z_cal))
            cal_rows.append(dict(method=name, param=g,
                                 far_per_1000=nb / (len(cal_idx) * 24) * 1000))
        print(f"FAR {name}: {time.time()-t0:.0f}с", flush=True)
    cal = pd.DataFrame(cal_rows)
    frozen = working_points(cal, FAR_BUDGET)
    print(f"рабочие точки (бюджет {FAR_BUDGET}/1000): {frozen}", flush=True)
    far_map = {(r.method, r.param): r.far_per_1000 for r in cal.itertuples()}

    # --- парные инъекции: детекты каждого ряда сохраняются → bootstrap по рядам
    plans = injection_plans(len(meas))
    rngb = np.random.default_rng(SEED_BOOT)
    rows = []
    for kind, d in CELLS:
        taus = plans[(kind, d)]
        # z и p_restart по рядам — один раз на ячейку
        Zs, Ps = [], []
        for j, i in enumerate(meas):
            y = W[i]
            s = mad_scale(y)
            z = ds(inject(y, kind, int(taus[j]), d, s=s), i)
            Zs.append(z)
            Ps.append(bocpd_prestart(z))
        for name, (_, fn, grid) in GRIDS.items():
            for g in grid:
                rec = np.zeros((len(meas), 3))
                delays = []
                for j in range(len(meas)):
                    found = detect_all(Zs[j], Ps[j], name, g)
                    tau = int(taus[j])
                    hits = [f for f in found if abs(f - tau) <= 2]
                    rec[j, 0] = 1 if hits else 0
                    rec[j, 1] = len([f for f in found if abs(f - tau) > 2])
                    rec[j, 2] = 0 if hits else 1
                    if hits:
                        delays.append(min(hits) - tau)
                prec, rec_, f1 = f1_of(rec)
                boots = [f1_of(rec[rngb.integers(0, len(meas), len(meas))])[2]
                         for _ in range(N_BOOT)]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                rows.append(dict(method=name, param=g, kind=kind, delta=d,
                                 far_per_1000=far_map[(name, g)],
                                 precision=prec, recall=rec_, f1=f1,
                                 f1_lo=lo, f1_hi=hi,
                                 delay=np.mean(delays) if delays else np.nan,
                                 is_working=(frozen[name] is not None
                                             and abs(g - frozen[name]) < 1e-12)))
            print(f"{kind} {d} {name}: {time.time()-t0:.0f}с", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "cpd_frontier_v2.csv", index=False)

    # дисплей: рабочие точки (FAR <= бюджет) и лучшее по F1 на level 1.6σ
    wd = out[out.is_working & (out.kind == "level")]
    print("\nРабочие точки, level-инъекции:", flush=True)
    print(wd.round(3).to_string(index=False), flush=True)
    print(f"\ntotal {time.time()-t0:.0f}с", flush=True)


if __name__ == "__main__":
    main()
