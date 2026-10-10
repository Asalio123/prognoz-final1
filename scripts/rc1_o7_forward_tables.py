"""Таблицы форвардной валидации 2025-H1 (задачи 1, 2, 4).

Ничего не перегоняет: читает готовые артефакты outputs/forecast_2025*.

Выходы:
- outputs/rc1_o7_forward_by_category.csv   — MAE (п.п. г/г) vs официальный ряд по
  категориям × месяцам янв–июн 2025 + флаг «внутри PI80». Строгий маппинг на
  официальные типы; категории без официального ряда помечены no_official_truth.
- outputs/rc1_o7_forward_monthly_aggregate.csv — помесячно «факт/прогноз/lo80/hi80/
  внутри полосы» для «Все категории» (готово для фан-чарта F4).
- outputs/rc1_o7_control_13_3.csv — контрольная сверка +13.3% г/г, пересчитанная
  напрямую из forecast_2025H1.parquet (агрегация суммой по МО).

Запуск: .venv/bin/python scripts/rc1_o7_forward_tables.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Строгий маппинг наша категория → официальный тип (как в scripts/28, CAT_MAP)
STRICT_MAP = {"Все категории": "Всего",
              "Продовольствие": "Продовольственные товары",
              "Общественное питание": "Общественное питание"}
ALL_CATS = ["Все категории", "Продовольствие", "Общественное питание",
            "Здоровье", "Маркетплейсы", "Транспорт"]
MONTHS_H1 = [f"2025-0{i}" for i in range(1, 7)]


def main():
    check = pd.read_csv(ROOT / "outputs" / "forecast_2025_check.csv")
    check = check[(check["fc_model"] == "DriftAdjSNaive") &
                  (check["month"].isin(MONTHS_H1))].copy()
    check["in_pi80"] = ((check["official_yoy_pct"] >= check["fc_yoy_lo80_pct"]) &
                        (check["official_yoy_pct"] <= check["fc_yoy_hi80_pct"]))
    check["abs_err_pp"] = (check["fc_yoy_pct"] - check["official_yoy_pct"]).abs()

    # --- Задача 1: таблица по категориям (длинный формат: строка = категория × месяц)
    rows = []
    for cat in ALL_CATS:
        sub = check[check["scope"] == cat]
        for m in MONTHS_H1:
            r = sub[sub["month"] == m]
            if cat not in STRICT_MAP:
                rows.append({"category": cat, "month": m, "official_type": "—",
                             "official_yoy_pct": np.nan, "fc_yoy_pct": np.nan,
                             "fc_lo80_pct": np.nan, "fc_hi80_pct": np.nan,
                             "abs_err_pp": np.nan, "in_pi80": "",
                             "note": "no_official_truth"})
                continue
            if not len(r):
                rows.append({"category": cat, "month": m, "official_type": STRICT_MAP[cat],
                             "official_yoy_pct": np.nan, "fc_yoy_pct": np.nan,
                             "fc_lo80_pct": np.nan, "fc_hi80_pct": np.nan,
                             "abs_err_pp": np.nan, "in_pi80": "", "note": "no_row"})
                continue
            r = r.iloc[0]
            rows.append({"category": cat, "month": m, "official_type": STRICT_MAP[cat],
                         "official_yoy_pct": r["official_yoy_pct"],
                         "fc_yoy_pct": r["fc_yoy_pct"],
                         "fc_lo80_pct": r["fc_yoy_lo80_pct"],
                         "fc_hi80_pct": r["fc_yoy_hi80_pct"],
                         "abs_err_pp": r["abs_err_pp"],
                         "in_pi80": "да" if r["in_pi80"] else "НЕТ",
                         "note": ""})
    by_cat = pd.DataFrame(rows)
    by_cat.to_csv(ROOT / "outputs" / "rc1_o7_forward_by_category.csv", index=False)

    # Сводка MAE / покрытие по категориям (только где есть официальный ряд)
    summ = (by_cat.dropna(subset=["abs_err_pp"])
                  .groupby("category")
                  .agg(official_type=("official_type", "first"),
                       mae_pp=("abs_err_pp", "mean"),
                       max_err_pp=("abs_err_pp", "max"),
                       in_pi80_n=("in_pi80", lambda s: (s == "да").sum()),
                       n_months=("month", "count"))
                  .reset_index())
    print("=== Задача 1: MAE по категориям (DriftAdjSNaive, 2025-H1, п.п. г/г) ===")
    print(summ.round(2).to_string(index=False))
    no_truth = [c for c in ALL_CATS if c not in STRICT_MAP]
    print(f"Без официального ряда (валидация невозможна): {', '.join(no_truth)}")

    # --- Задача 2: помесячная таблица агрегата для фан-чарта
    agg = (check[check["scope"] == "Все категории"]
           [["month", "official_yoy_pct", "fc_yoy_pct",
             "fc_yoy_lo80_pct", "fc_yoy_hi80_pct", "in_pi80"]]
           .rename(columns={"official_yoy_pct": "fact_yoy_pct",
                            "fc_yoy_pct": "forecast_yoy_pct",
                            "fc_yoy_lo80_pct": "lo80_yoy_pct",
                            "fc_yoy_hi80_pct": "hi80_yoy_pct"})
           .sort_values("month"))
    agg["in_pi80"] = agg["in_pi80"].map({True: "да", False: "НЕТ"})
    agg.to_csv(ROOT / "outputs" / "rc1_o7_forward_monthly_aggregate.csv", index=False)
    print("\n=== Задача 2: помесячно, «Все категории» ===")
    print(agg.round(2).to_string(index=False))
    print(f"MAE агрегат: {check[check['scope']=='Все категории']['abs_err_pp'].mean():.2f} п.п.; "
          f"в PI80: {(check[check['scope']=='Все категории']['in_pi80']).sum()}/6")

    # --- Задача 4: контроль +13.3% напрямую из forecast_2025H1.parquet
    fc = pd.read_parquet(ROOT / "outputs" / "forecast_2025H1.parquet")
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    act = (panel[panel["category"] == "Все категории"]
           .groupby("date")["value"].sum())
    act.index = pd.PeriodIndex(act.index, freq="M")
    fc_d = fc[(fc["model"] == "DriftAdjSNaive") & (fc["category"] == "Все категории")]
    fc_sum = fc_d.groupby("month")["y_pred"].sum()
    fc_sum.index = pd.PeriodIndex(fc_sum.index, freq="M")
    ctrl = []
    for m in pd.period_range("2025-01", "2025-06", freq="M"):
        prev = act.get(m - 12, np.nan)
        yoy = (fc_sum.get(m, np.nan) / prev - 1) * 100
        ctrl.append({"month": str(m), "fc_sum_rub": fc_sum.get(m, np.nan),
                     "actual_2024_rub": prev, "fc_yoy_pct": yoy})
    ctrl = pd.DataFrame(ctrl)
    # независимая проверка дрейфа: pooled медиана log-YoY 2024-12 vs 2023-12
    p = panel[panel["category"] == "Все категории"].copy()
    p["mi"] = (pd.PeriodIndex(p["date"], freq="M").year - 2023) * 12 + \
              pd.PeriodIndex(p["date"], freq="M").month - 1
    wide = p.pivot_table(index="territory_id", columns="mi", values="log_value")
    g = float(np.nanmedian((wide[23] - wide[11]).to_numpy()))
    ctrl.to_csv(ROOT / "outputs" / "rc1_o7_control_13_3.csv", index=False)
    print("\n=== Задача 4: контроль +13.3% из forecast_2025H1.parquet ===")
    print(ctrl.round(2).to_string(index=False))
    print(f"pooled drift g (log, пересчитан) = {g:.4f} → {(np.exp(g)-1)*100:.2f}% г/г")
    print(f"fc_yoy по месяцам: min {ctrl['fc_yoy_pct'].min():.2f}, "
          f"max {ctrl['fc_yoy_pct'].max():.2f} (константно по построению DriftAdjSNaive)")


if __name__ == "__main__":
    main()
