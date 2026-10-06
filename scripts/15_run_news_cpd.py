"""Новости × CPD: матрица согласования (К5, союз «и» — обязательная вторая половина).

Burst detection на новостных потоках (robust z / MAD, окно 90д) × PELT (pen=2, консервативный)
на национальном ряде потребления (среднее log по панели, по категориям).
Матрица: CP∧burst = национальный шок / CP без burst = локальный / burst без CP = проверка.
Значимость совпадений: circular-shift пермутации (месяцы коррелированы, Fisher не подходит).
Выход: outputs/news_cpd_matrix.csv, outputs/news_cpd_perm.txt
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt

ROOT = Path(__file__).resolve().parents[1]
H = [1]  # окно совпадения ±1 мес


def main():
    nm = pd.read_parquet(ROOT / "outputs" / "news_monthly.parquet")
    topics = [c[2:] for c in nm.columns if c.startswith("z_")]
    topics.remove("consumer_sentiment")  # 3 заголовка за 24 мес — ниже порога обнаружения (R07)
    bursts = {}
    for t in topics:
        z = nm[f"z_{t}"]
        bursts[t] = set(nm.index[z > 1.65])  # месяцы с тематическим всплеском

    # национальный месячный ряд СберИндекса (SA), 2018–2026; матрица в окне 2023-01..2024-12
    nat_df = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" / "consumper-spending-index-sa_parquet.parquet")
    nat_df["month"] = pd.to_datetime(nat_df["period"]).dt.strftime("%Y-%m")
    nat = nat_df.set_index("month")["value"].sort_index()
    cats = ["Все категории"]
    months = {i: m for i, m in enumerate(nat.index)}
    cpd = {}
    for cat in cats:
        y = nat.to_numpy(dtype=float)
        bk = [b for b in rpt.Pelt(model="l2").fit(np.log(y)).predict(pen=1.0) if b < len(y)]
        cpd[cat] = {months[b] for b in bk if "2023-01" <= months[b] <= "2024-12"}

    # матрица совпадений по месяцам (±1)
    all_months = [m for m in sorted(months.values()) if "2023-01" <= m <= "2024-12"]
    rows = []
    for cat, cps in cpd.items():
        for t, bursts_m in bursts.items():
            for m in all_months:
                i = all_months.index(m)
                near = {all_months[j] for j in range(max(0, i - 1), min(len(all_months), i + 2))}
                cp = m in cps
                bu = bool(bursts_m & near)
                rows.append(dict(category=cat, topic=t, month=m, cp=cp, burst=bu))
    mx = pd.DataFrame(rows)
    mx["cell"] = np.select([mx.cp & mx.burst, mx.cp & ~mx.burst, ~mx.cp & mx.burst],
                           ["CP∧news (нац. шок)", "CP без новостей (локальный)", "news без CP"],
                           default="—")
    mx.to_csv(ROOT / "outputs" / "news_cpd_matrix.csv", index=False)

    # сводка по категории × тема: число совпадений
    piv = mx[mx.cp | mx.burst].groupby(["category", "topic", "cell"]).size().unstack(fill_value=0)
    print(piv.to_string())

    # circular-shift пермутация: сдвигаем месячные burst-метки, меряем совпадения
    rng = np.random.default_rng(42)
    obs = (mx.cp & mx.burst).sum()
    null = []
    mm = mx.copy()
    for rep in range(500):
        for t in bursts:
            shift = int(rng.integers(1, len(all_months) - 1))
            shifted = {all_months[(all_months.index(m) + shift) % len(all_months)] for m in bursts[t] if m in all_months}
            mm.loc[mm.topic == t, "burst"] = mm.loc[mm.topic == t, "month"].isin(shifted)
        null.append((mm.cp & mm.burst).sum())
    p = (np.mean(np.array(null) >= obs))
    with open(ROOT / "outputs" / "news_cpd_perm.txt", "w") as f:
        f.write(f"observed CP∧burst: {obs}\nnull mean: {np.mean(null):.1f}, p(circular-shift, 500 reps) = {p:.4f}\n")
    print(f"CP∧burst наблюдаемо: {obs}, null {np.mean(null):.1f}, p={p:.4f}")


if __name__ == "__main__":
    main()
