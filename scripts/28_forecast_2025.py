"""Форвардный прогноз 2025 (янв–июнь + полный год SeasProfileHA) — продуктовый артефакт.

Каузальность: чемпионы refit на ВСЕХ 24 месяцах панели (2023-01..2024-12),
прогноз строится из origin 24 (train ≤ 2024-12), ничего из 2025 не используется
(официальный ряд СберИндекса 2025 участвует ТОЛЬКО в пост-факт сверке агрегата).

Модели (champion.yaml, methodology §2):
- DriftAdjSNaive — h=1..6 (2025-01..2025-06), g = pooled медиана YoY на конце панели.
- SeasProfileHA — полный 2025 (h=1..12 из origin 24): med_all + шринкаж-сезонность (k=2).

Интервалы: простые эмпирические (B3 не успел) — ±q80/q95 |log(y_true/y_pred)|
остатков чемпиона на backtest (preds_seasadj.parquet) по модели × горизонту ×
квинтилю масштаба ряда (медиана value за 2023–2024). Промежуточные горизонты
(h=2,4,5,7..11) — линейная интерполяция квантилей по h из {1,3,6,12}.

Сверка: нац. агрегат «Все категории» (сумма прогноза по МО) vs официальный
consumer-spending-growth (Номинальное/Всего, % г/г) за 2025-01..2025-06.
Это ЕДИНСТВЕННАЯ честная проверка: ground truth на уровне МО за 2025 нет.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAL = np.arange(24) % 12
H_DRIFT = [1, 2, 3, 4, 5, 6]      # DriftAdjSNaive → 2025-01..06
H_SEAS = list(range(1, 13))       # SeasProfileHA → весь 2025
BT_HORIZONS = np.array([1, 3, 6, 12])  # горизонты, где есть backtest-остатки


def load_wide():
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    meta = panel.groupby("series_id").agg(
        territory_id=("territory_id", "first"), category=("category", "first"),
        region_name=("region_name", "first"))
    return panel, wide, wide_rub, meta


def forward_forecasts(wv, wr, meta):
    """Refit на всех 24 месяцах (origin 24, c=23) → прогноз 2025."""
    c = 23
    train_cols = list(range(24))
    med_all = np.nanmedian(wv[:, train_cols], axis=1)
    g = float(np.nanmedian(wv[:, c] - wv[:, c - 12]))  # pooled YoY дрейф, 2024-12 vs 2023-12
    rows = []

    # DriftAdjSNaive h=1..6 → 2025-01..2025-06
    for h in H_DRIFT:
        tm = c + h  # 24..29
        p_log = wv[:, tm - 12] + g
        rows.append(pd.DataFrame({"series_id": meta.index, "month_idx": tm, "horizon": h,
                                  "y_pred": np.expm1(np.clip(p_log, 0, None)),
                                  "model": "DriftAdjSNaive"}))

    # SeasProfileHA h=1..12 → весь 2025
    reg = meta["region_name"]
    train_df = pd.DataFrame(wv[:, train_cols])
    reg_med_train = train_df.groupby(reg.values).median().median(axis=1)
    k_shrink = 2.0
    for h in H_SEAS:
        tm = c + h  # 24..35
        mcols = [i for i in train_cols if CAL[i] == tm % 12]  # всегда 2 (2023 + 2024)
        mo_seas = np.nanmedian(wv[:, mcols], axis=1) - med_all
        reg_ratio = (pd.DataFrame(wv[:, mcols]).groupby(reg.values).median().median(axis=1)
                     - reg_med_train).reindex(reg.values).to_numpy()
        seas = (len(mcols) * np.nan_to_num(mo_seas, nan=0.0) + k_shrink * reg_ratio) / (len(mcols) + k_shrink)
        p_log2 = med_all + seas
        rows.append(pd.DataFrame({"series_id": meta.index, "month_idx": tm, "horizon": h,
                                  "y_pred": np.expm1(np.clip(p_log2, 0, None)),
                                  "model": "SeasProfileHA"}))
    fc = pd.concat(rows, ignore_index=True)
    fc["month"] = pd.period_range("2025-01", periods=12, freq="M")[fc["month_idx"].to_numpy() - 24].astype(str)
    return fc, g


def empirical_intervals(fc, wr, meta):
    """q80/q95 |log-resid| по модели × горизонту × квинтилю масштаба; интерполяция по h."""
    bt = pd.read_parquet(ROOT / "outputs" / "preds_seasadj.parquet")
    bt = bt[(bt["y_true"] > 0) & (bt["y_pred"] > 0)].copy()
    bt["resid"] = np.abs(np.log(bt["y_true"] / bt["y_pred"]))
    # масштаб ряда = медиана value за 2023–2024 (фиксированный, in-sample)
    scale = pd.Series(np.nanmedian(wr, axis=1), index=meta.index, name="scale")
    quint = pd.qcut(scale, 5, labels=False, duplicates="drop").rename("scale_quintile")
    bt = bt.merge(quint, on="series_id", how="left")

    qs = (bt.groupby(["model", "horizon", "scale_quintile"])["resid"]
            .quantile([0.80, 0.95]).unstack().rename(columns={0.80: "q80", 0.95: "q95"}))

    def interp_q(model, quintile, h):
        """Линейная интерполяция квантилей между backtest-горизонтами {1,3,6,12}."""
        try:
            sub = qs.loc[model].xs(quintile, level="scale_quintile")
        except KeyError:
            sub = qs.loc[model].groupby(level="horizon").median()  # fallback: без квинтиля
        sub = sub.reindex(BT_HORIZONS)
        return (float(np.interp(h, BT_HORIZONS, sub["q80"].to_numpy())),
                float(np.interp(h, BT_HORIZONS, sub["q95"].to_numpy())))

    fc = fc.merge(quint, on="series_id", how="left")
    q = np.array([interp_q(m, int(qq), h) for m, qq, h in
                  zip(fc["model"], fc["scale_quintile"].fillna(2), fc["horizon"])])
    fc["y_lo80"] = fc["y_pred"] * np.exp(-q[:, 0])
    fc["y_hi80"] = fc["y_pred"] * np.exp(+q[:, 0])
    fc["y_lo95"] = fc["y_pred"] * np.exp(-q[:, 1])
    fc["y_hi95"] = fc["y_pred"] * np.exp(+q[:, 1])
    coverage_ref = (bt.groupby(["model", "horizon"])["resid"]
                      .apply(lambda r: pd.Series({
                          "cov80_emp": (r <= r.quantile(0.80)).mean(),
                          "n": len(r)})).unstack())
    return fc, qs, coverage_ref


CAT_MAP = {"Все категории": "Всего",  # наша категория → официальный ряд
           "Продовольствие": "Продовольственные товары",
           "Общественное питание": "Общественное питание"}


def national_check(fc, panel):
    """Наш агрегат YoY 2025 vs официальный номинальный % г/г СберИндекса.

    Scopes: «Все категории»→«Всего» + две маппируемые категории. Интервалы —
    сумма per-series границ (консервативно: предполагает полную корреляцию ошибок).
    Официальный ряд идёт до 2026-08 → сверяем ВЕСЬ 2025 (не только H1).
    """
    off = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" /
                          "consumer-spending-growth_parquet.parquet")
    off = off[off["value_type"] == "Номинальное"].copy()
    off["month"] = pd.PeriodIndex(off["period"], freq="M")
    off = off.pivot_table(index="month", columns="type", values="value")

    rows = []
    for cat, off_type in CAT_MAP.items():
        act = panel[panel["category"] == cat].groupby("date")["value"].sum()
        act.index = pd.PeriodIndex(act.index, freq="M")
        fc_cat = fc[fc["category"] == cat]
        fc_sum = fc_cat.groupby(["month", "model"]).agg(
            y_pred=("y_pred", "sum"), y_lo80=("y_lo80", "sum"), y_hi80=("y_hi80", "sum"),
            y_lo95=("y_lo95", "sum"), y_hi95=("y_hi95", "sum")).reset_index()
        fc_sum["month"] = pd.PeriodIndex(fc_sum["month"], freq="M")
        for m in pd.period_range("2024-01", "2025-12", freq="M"):
            prev = act.get(m - 12, np.nan)
            cur = act.get(m, np.nan)
            base = {"scope": cat, "month": str(m),
                    "panel_yoy_actual_pct": (cur / prev - 1) * 100 if prev and cur else np.nan,
                    "official_yoy_pct": float(off[off_type].get(m, np.nan))
                    if off_type in off.columns else np.nan}
            models = (["DriftAdjSNaive", "SeasProfileHA"] if cat == "Все категории"
                      else ["DriftAdjSNaive"]) if str(m) >= "2025-01" else []
            appended = False
            for model in models:
                r = fc_sum[(fc_sum["month"] == m) & (fc_sum["model"] == model)]
                if len(r) and prev:
                    r = r.iloc[0]
                    rows.append({**base, "fc_model": model,
                                 "fc_yoy_pct": (r["y_pred"] / prev - 1) * 100,
                                 "fc_yoy_lo80_pct": (r["y_lo80"] / prev - 1) * 100,
                                 "fc_yoy_hi80_pct": (r["y_hi80"] / prev - 1) * 100,
                                 "fc_yoy_lo95_pct": (r["y_lo95"] / prev - 1) * 100,
                                 "fc_yoy_hi95_pct": (r["y_hi95"] / prev - 1) * 100})
                    appended = True
            if not appended:  # 2024: только факт панели и официальный ряд (контекст трекинга)
                rows.append({**base, "fc_model": "", "fc_yoy_pct": np.nan,
                             "fc_yoy_lo80_pct": np.nan, "fc_yoy_hi80_pct": np.nan,
                             "fc_yoy_lo95_pct": np.nan, "fc_yoy_hi95_pct": np.nan})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    panel, wide, wide_rub, meta = load_wide()
    wv, wr = wide.to_numpy(), wide_rub.to_numpy()
    print(f"panel: {len(meta)} рядов, {meta['territory_id'].nunique()} МО, "
          f"{meta['category'].nunique()} категорий")

    fc, g = forward_forecasts(wv, wr, meta)
    print(f"pooled drift g (log) = {g:.4f} → YoY {(np.exp(g)-1)*100:.1f}%")

    fc, qs, cov = empirical_intervals(fc, wr, meta)
    fc = fc.merge(meta.reset_index()[["series_id", "territory_id", "category", "region_name"]],
                  on="series_id", how="left")
    out = fc[["series_id", "territory_id", "category", "region_name", "month", "horizon",
              "model", "y_pred", "y_lo80", "y_hi80", "y_lo95", "y_hi95", "scale_quintile"]]
    out.to_parquet(ROOT / "outputs" / "forecast_2025H1.parquet", index=False)
    print(f"saved outputs/forecast_2025H1.parquet: {len(out)} строк "
          f"({len(out)//12} рядов × 12 мес SeasProfileHA + {len(out[out.model=='DriftAdjSNaive'])//6} × 6)")

    qs.to_csv(ROOT / "outputs" / "forecast_2025_interval_quantiles.csv")
    print("\nКвантили |log-resid| (медиана по квинтилям):")
    print(qs.groupby(level=["model", "horizon"]).median().round(4).to_string())

    check = national_check(fc, panel)
    check.to_csv(ROOT / "outputs" / "forecast_2025_check.csv", index=False)
    print("\nНац. сверка (YoY, %):")
    print(check.round(2).to_string(index=False))
    print(f"\n{time.time()-t0:.0f}с")


if __name__ == "__main__":
    main()
