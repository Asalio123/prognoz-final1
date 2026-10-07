"""rc1 / O6 (бонус): nowcast-рамка — lag-корреляции новостных индексов с нац. рядом.

Вопрос: на каком горизонте тематические новостные индексы вообще согласованы с
национальным рядом потребления. ДИАГНОСТИКА механизма, не новое заявление о победе.

Ряды: z-индексы 4 активных тем (outputs/news_monthly_v3.parquet, 2023-01..2024-12,
24 мес) и национальный ряд consumer-spending-growth «Всего» Номинальное (тот же,
что в burst×CPD) на той же сетке месяцев.

Лаг l: corr(z_t, nat_{t+l}) — l>0: новости ПРЕДШЕСТВУЮТ ряду; l<0: отстают.
n = 24-|l| точек; двусторонний p по t-распределению. 20 корреляций на тему-лаг —
множественная проверка: Holm поверх p. Осторожность: 24 точки, мощность мизерная.

Выход: outputs/rc1_o6_nowcast_lags.csv, outputs/rc1_o6_nowcast_lags.md
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
LAGS = [-2, -1, 0, 1, 2]


def main():
    nm = pd.read_parquet(ROOT / "outputs" / "news_monthly_v3.parquet")
    nm.index = nm.index.astype(str)
    topics = [c[2:] for c in nm.columns if c.startswith("z_")]

    g = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" / "consumer-spending-growth_parquet.parquet")
    g["month"] = pd.to_datetime(g["period"]).dt.strftime("%Y-%m")
    nat = g[(g["type"] == "Всего") & (g["value_type"] == "Номинальное")].set_index("month")["value"].sort_index()
    nat = nat[~nat.index.duplicated()]
    nat_w = nat.loc["2023-01":"2024-12"]
    assert len(nat_w) == 24 and len(nm) == 24

    rows = []
    for t in topics:
        z = nm[f"z_{t}"]
        for l in LAGS:
            zs = z.shift(-l) if l >= 0 else z.shift(-l)  # z_t vs nat_{t+l}
            pair = pd.concat([z, nat_w.shift(-l)], axis=1, keys=["z", "nat"]).dropna()
            r, p = stats.pearsonr(pair["z"], pair["nat"])
            rs, ps = stats.spearmanr(pair["z"], pair["nat"])
            rows.append(dict(topic=t, lag=l, n=len(pair), pearson_r=round(r, 3), p=round(p, 4),
                             spearman_r=round(rs, 3), p_spearman=round(ps, 4)))
    df = pd.DataFrame(rows)

    # Holm по 20 тестам (pearson p)
    order = df["p"].sort_values().index
    m = len(df)
    holm = pd.Series(index=order, dtype=float)
    prev = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * df.loc[idx, "p"])
        prev = max(prev, adj)
        holm[idx] = prev
    df["p_holm"] = holm.round(4)
    df.to_csv(ROOT / "outputs" / "rc1_o6_nowcast_lags.csv", index=False)

    piv = df.pivot(index="topic", columns="lag", values="pearson_r")
    md = ["# Nowcast-диагностика: lag-корреляции новостных z-индексов с нац. рядом (rc1/O6)", "",
          "z-индексы (news_monthly_v3) vs consumer-spending-growth «Всего» номин., 2023-01..2024-12 (24 мес).",
          "lag l: corr(z_t, nat_{t+l}); l>0 — новости предшествуют ряду. Pearson, n=24−|l|.",
          "20 тестов → Holm-поправка. Это диагностика механизма, НЕ заявление о качестве прогноза.", "",
          "| тема | lag −2 | lag −1 | lag 0 | lag +1 | lag +2 |", "|---|---|---|---|---|---|"]
    for t in topics:
        md.append(f"| {t} | " + " | ".join(f"{piv.loc[t, l]:+.3f}" for l in LAGS) + " |")
    md += ["", "p (сырые / Holm):", "", "| тема | lag | r | n | p | p_holm |", "|---|---|---|---|---|---|"]
    for r in df.itertuples():
        md.append(f"| {r.topic} | {r.lag:+d} | {r.pearson_r:+.3f} | {r.n} | {r.p:.4f} | {r.p_holm:.4f} |")
    md += ["", "## Чтение", ""]
    (ROOT / "outputs" / "rc1_o6_nowcast_lags.md").write_text("\n".join(md))
    print(piv.to_string())
    print()
    print(df[df.p < 0.10].to_string())


if __name__ == "__main__":
    main()
