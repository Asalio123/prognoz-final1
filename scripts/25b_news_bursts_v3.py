"""Дневные burst'ы новостных тем (К5, стыковка с CPD — R07 §4).

Детектор: robust z-score по дневным счётчикам темы —
    z_d = (x_d - median(W_d)) / scale_d,  scale_d = max(1.4826 * MAD(W_d), 1.0),
где W_d = счётчики предыдущих 90 дней (строго прошлое: окно [d-90 .. d-1],
min_periods=30) -> легально на любом origin'е, нормировка только назад.

Порог: burst-день при z > 3; burst-событие = run >= 2 подряд идущих burst-дней.
Событие присваивается месяцу ДАТЫ ПОДТВЕРЖДЕНИЯ (день, когда run достиг длины 2) —
момент, когда событие становится детектируемым; run на границе месяцев не дублируется.

Про scale-флор 1.0: на разреженных темах (rate_credit ~0.1 заголовка/день) медиана
и MAD окна равны 0 -> z был бы inf/NaN на любой ненулевой день. Флор = одно событие
на счётной шкале, константа предзадана (не оценивается по данным). При нём z>3
требует >= 4 заголовков за день над нулевой медианой — консервативно.

Вход:  outputs/news_daily_v2.parquet (4 активные темы, таксономия v3).
Выход: outputs/news_bursts_v3.csv — сетка месяц x тема (24 x 4 = 96 строк):
       month, topic, burst (0/1), n_burst_days, max_run_len, max_z, first_run_date.
       Пересчёт матрицы burst x CPD — НЕ этот скрипт (зона 15_run_news_cpd.py).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TOPICS = ["prices_inflation", "rate_credit", "sanctions_shocks", "emergencies_regional"]
WINDOW = 90
MIN_PERIODS = 30
Z_THR = 3.0
SCALE_FLOOR = 1.0


def robust_z_trailing(x: pd.Series) -> pd.Series:
    """z_d по окну из 90 предыдущих дней (x.shift(1) -> rolling(90))."""
    vals = x.to_numpy(dtype=float)
    z = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        lo = max(0, i - WINDOW)
        w = vals[lo:i]  # строго прошлые дни
        w = w[~np.isnan(w)]
        if len(w) < MIN_PERIODS:
            continue
        med = np.median(w)
        mad = np.median(np.abs(w - med))
        scale = max(1.4826 * mad, SCALE_FLOOR)
        z[i] = (vals[i] - med) / scale
    return pd.Series(z, index=x.index)


def runs_to_months(is_burst: pd.Series, z: pd.Series):
    """Из булева ряда burst-дней -> события (run>=2), присвоенные месяцу подтверждения.
    Возвращает (events: list[dict], stats: dict)."""
    b = is_burst.to_numpy()
    dates = is_burst.index
    events, i = [], 0
    while i < len(b):
        if b[i]:
            j = i
            while j + 1 < len(b) and b[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len >= 2:
                confirm_date = dates[i + 1]  # день, когда run достиг длины 2
                events.append(dict(confirm_month=confirm_date.strftime("%Y-%m"),
                                   start=dates[i], end=dates[j], run_len=run_len,
                                   max_z=float(z.iloc[i:j + 1].max())))
            i = j + 1
        else:
            i += 1
    stats = dict(n_burst_days=int(b.sum()),
                 max_run_len=max((e["run_len"] for e in events), default=0),
                 max_z=float(z.max()) if len(z) else np.nan)
    return events, stats


def main():
    daily = pd.read_parquet(ROOT / "outputs" / "news_daily_v2.parquet")
    daily.index = pd.to_datetime(daily.index).normalize()
    daily = daily[TOPICS]
    months = pd.period_range("2023-01", "2024-12", freq="M").astype(str)

    grid, all_events = [], []
    for t in TOPICS:
        z = robust_z_trailing(daily[t])
        is_burst = z > Z_THR
        events, stats = runs_to_months(is_burst, z)
        by_month = {}
        for e in events:
            e["topic"] = t
            by_month.setdefault(e["confirm_month"], e)
            all_events.append(e)
        for m in months:
            e = by_month.get(m)
            grid.append(dict(month=m, topic=t, burst=int(e is not None),
                             n_burst_days=int(is_burst[m].sum()),
                             max_run_len=e["run_len"] if e else 0,
                             max_z=round(float(z[m].max()), 3) if len(z[m]) else np.nan,
                             first_run_date=str(e["start"].date()) if e else ""))
        print(f"{t}: burst-дней {stats['n_burst_days']}, событий (run>=2) {len(events)}, "
              f"max z {stats['max_z']:.1f}")

    out = pd.DataFrame(grid)
    out.to_csv(ROOT / "outputs" / "news_bursts_v3.csv", index=False)
    print(f"\nsaved outputs/news_bursts_v3.csv: {len(out)} строк, "
          f"burst-месяцев {out['burst'].sum()} из {len(out)}")
    ev = pd.DataFrame(all_events).sort_values(["confirm_month", "topic"])
    print("\n=== события (месяц подтверждения | тема | run | даты | max z) ===")
    for e in ev.itertuples():
        print(f"  {e.confirm_month} | {e.topic:22s} | run={e.run_len} | "
              f"{e.start.date()}..{e.end.date()} | z={e.max_z:.1f}")


if __name__ == "__main__":
    main()
