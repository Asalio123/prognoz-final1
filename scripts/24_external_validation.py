"""24_external_validation.py — внешняя валидация панели против официальных рядов СберИндекса.

Пять блоков (DEEP_B4):
 1. Сверка агрегата: панель (2016 МО, «Все категории») vs нац. помесячный ряд 2023–2024.
 2. Реальный рост: номинал панели, дефлированный ИПЦ, vs официальный реальный ряд.
 3. Восстановление таблицы R05 §3.2 (нац. % г/г окт-2024…март-2025) из parquet.
 4. Ретро-проверка «радар на 2024-12»: онсеты CPD + новостные индексы vs факт янв–фев-2025.
 5. Разрыв эталона: помесячный vs недельный «Общепит», янв-2025.

Выходы: outputs/external_*.csv. Запуск: nice -n 10 .venv/bin/python scripts/24_external_validation.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXT = Path("/Users/said/сбериндекс/data/external")
API = EXT / "sberindex_api"
OUT = ROOT / "outputs"

panel = pd.read_parquet("/Users/said/сбериндекс/data/panel/panel.parquet",
                        columns=["territory_id", "category", "date", "value"])
panel["date"] = pd.PeriodIndex(panel["date"].astype(str), freq="M")

growth = pd.read_parquet(API / "consumer-spending-growth_parquet.parquet")
growth["period"] = pd.PeriodIndex(growth["period"].astype(str), freq="M")
levels = pd.read_parquet(API / "consumer-spending_parquet.parquet")
levels["period"] = pd.PeriodIndex(levels["period"].astype(str), freq="M")
weekly = pd.read_parquet(API / "ver-izmenenie-trat-po-kategoriyam_parquet.parquet")
weekly["period"] = pd.to_datetime(weekly["period"])

cpi_raw = pd.read_csv(EXT / "cbr_cpi_yoy_2023_2024.csv", sep=";")
cpi_raw["cpi_yoy"] = (cpi_raw["cpi_yoy"].astype(str)
                      .str.replace(" ", "", regex=False).str.replace(",", ".", regex=False)
                      .astype(float))
cpi_raw["date"] = pd.PeriodIndex(
    pd.to_datetime(cpi_raw["month"].astype(str).str.strip(), format="%m.%Y"), freq="M")
cpi = cpi_raw.set_index("date")["cpi_yoy"].sort_index()


def official_growth(value_type, cat):
    m = growth[(growth["value_type"] == value_type) & (growth["type"] == cat)]
    return m.set_index("period")["value"].sort_index()


# ---------- 1. Сверка агрегата ----------
allcat = panel[panel["category"] == "Все категории"]
agg = allcat.groupby("date")["value"].agg(panel_sum="sum", panel_median="median")
agg["yoy_sum"] = agg["panel_sum"].pct_change(12) * 100
agg["yoy_median"] = agg["panel_median"].pct_change(12) * 100

off_nom = official_growth("Номинальное", "Всего")
off_lvl = levels[levels["type"] == "Всего"].set_index("period")["value"].sort_index()

y24 = pd.period_range("2024-01", "2024-12", freq="M")
cmp1 = pd.DataFrame({
    "panel_yoy_sum_pct": agg["yoy_sum"],
    "panel_yoy_median_pct": agg["yoy_median"],
    "official_nominal_yoy_pct": off_nom,
}).loc[y24]

r_sum = cmp1["panel_yoy_sum_pct"].corr(cmp1["official_nominal_yoy_pct"])
r_med = cmp1["panel_yoy_median_pct"].corr(cmp1["official_nominal_yoy_pct"])
rho_sum = cmp1["panel_yoy_sum_pct"].corr(cmp1["official_nominal_yoy_pct"], method="spearman")
mae_sum = (cmp1["panel_yoy_sum_pct"] - cmp1["official_nominal_yoy_pct"]).abs().mean()
mae_med = (cmp1["panel_yoy_median_pct"] - cmp1["official_nominal_yoy_pct"]).abs().mean()

# уровни: 24 мес, панель-сумма (млн? руб·МО-ед.) vs офиц. млрд руб — масштабы разные, r масштабо-инвариантен
lvl = pd.DataFrame({"panel_sum": agg["panel_sum"], "official_lvl": off_lvl}).dropna()
r_lvl = lvl["panel_sum"].corr(lvl["official_lvl"])
r_lvl_logdiff = np.log(lvl["panel_sum"]).diff().corr(np.log(lvl["official_lvl"]).diff())

cmp1.to_csv(OUT / "external_panel_vs_official.csv")
print("=== 1. Агрегат: панель vs официальный номинальный ряд, 2024 ===")
print(cmp1.round(2).to_string())
print(f"r(уровни YoY, sum)   = {r_sum:.3f}   Spearman = {rho_sum:.3f}")
print(f"r(уровни YoY, median)= {r_med:.3f}")
print(f"MAE расхождения YoY: sum {mae_sum:.2f} п.п., median {mae_med:.2f} п.п.")
print(f"r(уровни рядов, 24 мес) = {r_lvl:.4f};  r(log-приросты м/м) = {r_lvl_logdiff:.3f}")
print(f"Годовой рост 2024/2023 по панели: сумма {agg['panel_sum'].loc['2024-01':'2024-12'].sum()/agg['panel_sum'].loc['2023-01':'2023-12'].sum()-1:+.4f}, "
      f"медианный ряд {agg['panel_median'].loc['2024-01':'2024-12'].sum()/agg['panel_median'].loc['2023-01':'2023-12'].sum()-1:+.4f}")

# ---------- 2. Реальный рост ----------
off_real = official_growth("Реальное", "Всего")
real = pd.DataFrame({
    "panel_nom_yoy": agg["yoy_sum"],
    "cpi_yoy": cpi,
    "official_real_yoy": off_real,
}).loc[y24]
real["panel_real_yoy"] = ((1 + real["panel_nom_yoy"] / 100) / (1 + real["cpi_yoy"] / 100) - 1) * 100
real.to_csv(OUT / "external_real_growth.csv")

nom_year = agg["panel_sum"].loc["2024-01":"2024-12"].sum() / agg["panel_sum"].loc["2023-01":"2023-12"].sum() - 1
cpi_avg24, cpi_avg23 = cpi.loc[y24].mean(), cpi.loc[pd.period_range("2023-01", "2023-12", freq="M")].mean()
real_year_avgcpi = (1 + nom_year) / (1 + cpi_avg24 / 100) - 1
r_real = real["panel_real_yoy"].corr(real["official_real_yoy"])
mae_real = (real["panel_real_yoy"] - real["official_real_yoy"]).abs().mean()
print("\n=== 2. Реальный рост, 2024 ===")
print(real.round(2).to_string())
print(f"Номинальный годовой рост панели: {nom_year*100:+.2f}%  (офиц. ном. среднее 2024: {off_nom.loc[y24].mean():.2f}%)")
print(f"ИПЦ: средний 2024 = {cpi_avg24:.2f}%, средний 2023 = {cpi_avg23:.2f}%")
print(f"Реальный рост панели (год, ср. ИПЦ): {real_year_avgcpi*100:+.2f}%")
print(f"Реальный рост панели помесячно: min {real['panel_real_yoy'].min():.2f}, max {real['panel_real_yoy'].max():.2f}, mean {real['panel_real_yoy'].mean():.2f}")
print(f"Офиц. реальный ряд 2024: min {off_real.loc[y24].min():.2f}, max {off_real.loc[y24].max():.2f}, mean {off_real.loc[y24].mean():.2f}")
print(f"r(panel_real, official_real) = {r_real:.3f};  MAE = {mae_real:.2f} п.п.")

# ---------- 3. Таблица R05 §3.2 ----------
win = pd.period_range("2024-10", "2025-03", freq="M")
cats = {"Всего": "Всего", "Продовольственные товары": "Продовольствие",
        "Непродовольственные товары": "Непродовольствие", "Услуги": "Услуги"}
tab = pd.DataFrame({short: official_growth("Номинальное", long).loc[win] for long, short in cats.items()})
tab_real = pd.DataFrame({short: official_growth("Реальное", long).loc[win] for long, short in cats.items()})
tab.to_csv(OUT / "external_national_2024_25.csv")
tab_real.to_csv(OUT / "external_national_2024_25_real.csv")
print("\n=== 3. Нац. номинальный рост % г/г (восстановление R05 §3.2) ===")
print(tab.round(1).to_string())
print("--- реальный ---")
print(tab_real.round(1).to_string())

# ---------- 4. Ретро-проверка радара на 2024-12 ----------
cpd = pd.read_parquet(OUT / "cpd_breaks_v2.parquet")
cpd["date"] = cpd["break_month"].map(lambda m: pd.Period("2023-01", freq="M") + int(m))
n_mo = panel["territory_id"].nunique()
onset_share = (cpd.groupby("date")["territory_id"].nunique() / n_mo * 100).rename("mo_with_break_pct")
# доля МО с брейком за скользящие 3 мес
roll3 = {}
for t in pd.period_range("2024-06", "2024-12", freq="M"):
    w = cpd[(cpd["date"] >= t - 2) & (cpd["date"] <= t)]
    roll3[t] = w["territory_id"].nunique() / n_mo * 100
roll3 = pd.Series(roll3, name="mo_with_break_3m_pct")

news = pd.read_parquet(OUT / "news_monthly_v2.parquet")
news_ctx = news.loc[["2024-10", "2024-11", "2024-12"],
                    ["count_rate_credit", "count_prices_inflation", "z_rate_credit", "z_prices_inflation"]]

fact = pd.DataFrame({
    "official_nom_yoy": off_nom.loc[pd.period_range("2024-10", "2025-03", freq="M")],
    "official_real_yoy": off_real.loc[pd.period_range("2024-10", "2025-03", freq="M")],
})
retro = pd.concat([onset_share, roll3], axis=1).join(fact, how="outer")
retro.to_csv(OUT / "external_radar_retro.csv")
news_ctx.to_csv(OUT / "external_radar_retro_news.csv")
print("\n=== 4. Ретро-проверка радара на 2024-12 ===")
print(retro.round(2).to_string())
print(news_ctx.round(2).to_string())

# ---------- 5. Разрыв эталона: Общепит ----------
ob_month_nom = official_growth("Номинальное", "Общественное питание")
ob_month_real = official_growth("Реальное", "Общественное питание")
ob_week = weekly[weekly["category"] == "Общественное питание"].set_index("period")["value"].sort_index()
ob_week_jan25 = ob_week.loc["2025-01-01":"2025-01-31"]
ob_week_dec24 = ob_week.loc["2024-12-01":"2024-12-31"]

gap = pd.DataFrame({
    "monthly_nominal_yoy": ob_month_nom.loc[pd.period_range("2024-10", "2025-03", freq="M")],
    "monthly_real_yoy": ob_month_real.loc[pd.period_range("2024-10", "2025-03", freq="M")],
    "weekly_mean_yoy_same_month": ob_week.groupby(ob_week.index.to_period("M")).mean()
        .loc[pd.period_range("2024-10", "2025-03", freq="M")],
})
gap["gap_pp"] = gap["monthly_nominal_yoy"] - gap["weekly_mean_yoy_same_month"]
gap.to_csv(OUT / "external_obshchepit_gap.csv")
print("\n=== 5. Разрыв эталона: Общепит, помесячно vs недельно ===")
print(gap.round(2).to_string())
print("Недельные янв-2025:", ob_week_jan25.round(1).tolist(), " mean =", round(ob_week_jan25.mean(), 2))
print("Недельные дек-2024:", ob_week_dec24.round(1).tolist(), " mean =", round(ob_week_dec24.mean(), 2))
print(f"Янв-2025: помесячный номинал {ob_month_nom.loc[pd.Period('2025-01')]:.1f}% vs недельный mean {ob_week_jan25.mean():.1f}% "
      f"→ разрыв {ob_month_nom.loc[pd.Period('2025-01')] - ob_week_jan25.mean():.1f} п.п.")
print(f"Дек-2024: помесячный номинал {ob_month_nom.loc[pd.Period('2024-12')]:.1f}% vs недельный mean {ob_week_dec24.mean():.1f}%")
