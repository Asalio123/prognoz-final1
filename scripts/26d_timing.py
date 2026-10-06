"""Замер фактического времени CPD-детекторов на всей панели (R04 §3, «время на панель —
замер, не оценка»; таблица стоимости для К3).

Протокол: вся панель (2016 МО × 6 категорий, фактическое число рядов из panel.parquet),
единый вход — десезонированный лог-ряд (src.cpd, общая предобработка замеряется отдельной
строкой shared_preprocessing: она одна на всех). Каждый детектор — на замороженной рабочей
точке cpd_calibration_v2.csv (как в проде); дополнительно строка PELT-BIC (эталон числа
брейков в matched-протоколе 26) и ClaSP (sktime 1.2, period_length=6, n_cps=1 — как в 26c).
Wall-clock, time.perf_counter, один процесс, nice -n 10. На машине параллельно могут идти
lgbm-джобы — замеры wall-clock это учитывают как честную операционную среду.

Выход: outputs/cpd_timing_v3.csv (method, param, n_series, total_sec, ms_per_series).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.cpd import (category_profiles, deseasonalize, bocpd_prestart, load_wide,  # noqa: E402
                     GRIDS, K_SHRINK, DETREND, ROBUST)


def main():
    wide, cats = load_wide(ROOT)
    W = wide.to_numpy()
    n = len(W)
    print(f"панель: {n} рядов", flush=True)

    cal = pd.read_csv(ROOT / "outputs" / "cpd_calibration_v2.csv")
    frozen = {r.method: float(r.param) for r in cal[cal.is_working].itertuples()}

    rows = []

    # общая предобработка (профили категорий + десезонирование всех рядов)
    t0 = time.perf_counter()
    profs = category_profiles(W, cats)
    prof_arr = np.array([profs[c] for c in cats])
    Z = np.vstack([deseasonalize(W[i], prof_arr[i], k=K_SHRINK, detrend=DETREND, robust=ROBUST)
                   for i in range(n)])
    dt = time.perf_counter() - t0
    rows.append(dict(method="shared_preprocessing", param=np.nan, n_series=n,
                     total_sec=dt, ms_per_series=dt / n * 1000))
    print(f"preprocessing: {dt:.1f}с", flush=True)

    def bench(name, fn):
        t0 = time.perf_counter()
        for i in range(n):
            fn(Z[i])
        dt = time.perf_counter() - t0
        rows.append(dict(method=name, param=frozen.get(name, np.nan), n_series=n,
                         total_sec=dt, ms_per_series=dt / n * 1000))
        print(f"{name}: {dt:.1f}с ({dt/n*1000:.3f} мс/ряд)", flush=True)

    for m in ["PELT", "BinSeg", "Window", "CUSUM", "EWMA"]:
        bench(m, lambda z, m=m: GRIDS[m][1](z, frozen[m]))
    def bocpd_det(z):
        p = bocpd_prestart(z)
        return [b for b in range(1, len(z)) if p[b] > frozen["BOCPD"]]
    bench("BOCPD", bocpd_det)

    # PELT-BIC (эталон matched-протокола): pen=2·log(24) на z/level_scale
    from src.cpd import level_scale
    import ruptures as rpt

    def pelt_bic(z):
        zs = z / level_scale(z)
        return [b for b in rpt.Pelt(model="l2", min_size=3).fit(zs).predict(pen=2 * np.log(24))
                if b < len(z)]
    t0 = time.perf_counter()
    for i in range(n):
        pelt_bic(Z[i])
    dt = time.perf_counter() - t0
    rows.append(dict(method="PELT_BIC", param=round(2 * np.log(24), 4), n_series=n,
                     total_sec=dt, ms_per_series=dt / n * 1000))
    print(f"PELT_BIC: {dt:.1f}с", flush=True)

    # ClaSP (как в 26c)
    try:
        from sktime.detection.clasp import ClaSPSegmentation
        t0 = time.perf_counter()
        for i in range(n):
            ClaSPSegmentation(period_length=6, n_cps=1).fit_predict(np.asarray(Z[i], float))
        dt = time.perf_counter() - t0
        rows.append(dict(method="ClaSP", param="period=6,n_cps=1", n_series=n,
                         total_sec=dt, ms_per_series=dt / n * 1000))
        print(f"ClaSP: {dt:.1f}с", flush=True)
    except ImportError as e:
        rows.append(dict(method="ClaSP", param="SKIPPED", n_series=0,
                         total_sec=np.nan, ms_per_series=np.nan))
        print(f"ClaSP пропущен: {e}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "cpd_timing_v3.csv", index=False)
    print(out.round(3).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
