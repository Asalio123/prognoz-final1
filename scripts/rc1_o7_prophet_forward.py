"""Prophet-B* на форварде 2025-H1 (задача 3).

Prophet НЕ гоняется по панели. Только национальный агрегат: 6 категорий,
сумма по МО, 24 месяца train (2023-01..2024-12) → прогноз 2025-01..06.
Конфиг = Prophet-B* из scripts/03 (yearly_seasonality=4, multiplicative,
changepoint_prior_scale=0.05): train-окно 24 мес ≥ 2 полных цикла → ветка B
легальна (origin-12 ловушка здесь не активна).

Сравнение: YoY прогноза (прогноз суммы 2025 / факт суммы 2024) против того же
официального ряда Лаборатории, что и для чемпиона DriftAdjSNaive.

Выход: outputs/rc1_o7_prophet_forward.csv — по категориям × месяцам:
prophet_yoy_pct, champion_yoy_pct, official_yoy_pct, abs_err обеих моделей.
Запуск: nice -n 10 .venv/bin/python scripts/rc1_o7_prophet_forward.py
"""
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ["CMDSTAN_VERBOSE"] = "0"
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

STRICT_MAP = {"Все категории": "Всего",
              "Продовольствие": "Продовольственные товары",
              "Общественное питание": "Общественное питание"}
MONTHS_H1 = pd.period_range("2025-01", "2025-06", freq="M")
PROPHET_B_CFG = dict(yearly_seasonality=4, seasonality_mode="multiplicative",
                     changepoint_prior_scale=0.05)


def fit_forward(nat: pd.DataFrame, category: str) -> pd.Series:
    """Prophet-B на 24 мес нац. агрегата → прогноз yhat на 2025-01..06 (уровень, ₽)."""
    from prophet import Prophet
    s = nat[nat["category"] == category].sort_values("ds")
    m = Prophet(**PROPHET_B_CFG)
    m.fit(s[["ds", "y"]])
    fut = m.make_future_dataframe(periods=6, freq="MS")
    pr = m.predict(fut)
    out = pr.tail(6).set_index(pd.PeriodIndex(pr.tail(6)["ds"], freq="M"))["yhat"]
    return out.clip(lower=0)


def main():
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["ds"] = pd.PeriodIndex(panel["date"], freq="M").to_timestamp()
    nat = (panel.groupby(["category", "ds"])["value"].sum().reset_index()
           .rename(columns={"value": "y"}))
    nat24 = nat[nat["ds"] <= "2024-12-01"]  # train: строго ≤ 2024-12

    # официальный ряд (номинал) + факт нац. агрегата 2024 (база YoY)
    off = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" /
                          "consumer-spending-growth_parquet.parquet")
    off = off[off["value_type"] == "Номинальное"]
    off["month"] = pd.PeriodIndex(off["period"], freq="M")
    off = off.pivot_table(index="month", columns="type", values="value")

    # чемпион: готовые числа форварда (scripts/28), без перегона
    check = pd.read_csv(ROOT / "outputs" / "forecast_2025_check.csv")
    champ = (check[(check["fc_model"] == "DriftAdjSNaive")]
             .set_index(["scope", "month"]))

    rows = []
    for cat in sorted(nat24["category"].unique()):
        fc = fit_forward(nat24, cat)
        s = nat24[nat24["category"] == cat].set_index(pd.PeriodIndex(
            nat24[nat24["category"] == cat]["ds"], freq="M"))["y"]
        for m in MONTHS_H1:
            prev = s.get(m - 12, np.nan)
            p_yoy = (fc.get(m, np.nan) / prev - 1) * 100 if prev else np.nan
            official = float(off[STRICT_MAP[cat]].get(m, np.nan)) \
                if cat in STRICT_MAP else np.nan
            c = champ.loc[(cat, str(m))] if (cat, str(m)) in champ.index else None
            c_yoy = float(c["fc_yoy_pct"]) if c is not None else np.nan
            rows.append({
                "category": cat,
                "month": str(m),
                "official_type": STRICT_MAP.get(cat, "—"),
                "official_yoy_pct": official,
                "prophet_yoy_pct": p_yoy,
                "champion_yoy_pct": c_yoy,
                "prophet_abs_err_pp": abs(p_yoy - official)
                    if np.isfinite(official) and np.isfinite(p_yoy) else np.nan,
                "champion_abs_err_pp": abs(c_yoy - official)
                    if np.isfinite(official) and np.isfinite(c_yoy) else np.nan,
            })
        print(f"{cat}: Prophet YoY 2025-H1 = "
              f"{np.nanmean([r['prophet_yoy_pct'] for r in rows[-6:]]):.2f}% (ср.)")

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "outputs" / "rc1_o7_prophet_forward.csv", index=False)

    print("\n=== Prophet-B* форвард vs чемпион (MAE п.п. против официального ряда) ===")
    summ = (df.dropna(subset=["official_yoy_pct"])
              .groupby("category")
              .agg(official_type=("official_type", "first"),
                   prophet_mae_pp=("prophet_abs_err_pp", "mean"),
                   champion_mae_pp=("champion_abs_err_pp", "mean"),
                   n=("month", "count"))
              .reset_index())
    summ["delta_pp (prophet − champion)"] = summ["prophet_mae_pp"] - summ["champion_mae_pp"]
    print(summ.round(2).to_string(index=False))
    no_truth = sorted(set(df["category"]) - set(STRICT_MAP))
    print(f"\nБез официального ряда (только прогноз, без MAE): {', '.join(no_truth)}")


if __name__ == "__main__":
    main()
