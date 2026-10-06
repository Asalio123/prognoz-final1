"""T5: таблица «условия применения» (R02 Т5) + проверка правила чемпиона (champion.yaml).

Часть A — T5_conditions.csv: для топ-8 моделей (union топ-5 по pooled RelMAE каждого
горизонта из T1) по каждому из 4 разрезов (категория, масштаб, тип МО, волатильность):
лучшая/худшая ячейка (slice × горизонт) по RelMAE + текстовая формулировка.
Источник: outputs/T2_slice_{category,scale,motype,volatility}.csv — только вычисленные
числа, никакой литературы.

Часть B — T5_champion_rule.csv: правило no_failing_slice из configs/champion.yaml.
 1) Отбраковка: mae_ratio_vs_prophet_b < 1.0 на co-primary горизонтах [1, 3]
    (считается в двух рамках: pooled по 6 категориям = T1 и headline «Все категории» = T2).
 2) Для выживших: min win rate против Prophet-B* по всем значениям 4 разрезов
    (cell-level, как T1b; и per-series, как R02 §2.6). Чемпион = максимум минимума.
Win rate считается на тех же matched cells, что и T1/T2/T3 (пересечение 20 моделей).

Запуск: nice -n 10 .venv/bin/python scripts/21_t5_conditions.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
CO_PRIMARY = [1, 3]  # configs/validation.yaml
SLICES = [("category", "category"), ("scale", "scale_q"), ("motype", "mtype"), ("volatility", "vol_t")]
# файлы, исключённые из matched-геометрии T1 (04_build_tables.py + сиды LGBM, появившиеся
# после сборки T1 — их включение сместило бы пересечение относительно опубликованных таблиц)
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*", "preds_chronos2_quant_o12*",
                "preds_chronos2_zs_raw_o12*", "preds_coldstart*", "preds_lgbm_s*"]


def top8_models() -> list[str]:
    t1 = pd.read_csv(ROOT / "outputs" / "T1_model_x_horizon.csv")
    top = []
    for h in HORIZONS:
        top += t1[t1.horizon == h].nsmallest(5, "relmae_vs_snaive")["model"].tolist()
    seen, out = set(), []
    for m in top:
        if m not in seen:
            seen.add(m)
            out.append(m)
    assert len(out) == 8, out
    return out


def part_a_conditions(models: list[str]) -> pd.DataFrame:
    frames = {}
    for name, _ in SLICES:
        frames[name] = pd.read_csv(ROOT / "outputs" / f"T2_slice_{name}.csv")
    slice_col = {"category": "category", "scale": "scale_q", "motype": "mtype", "volatility": "vol_t"}
    rows = []
    for m in models:
        for sname, scol in slice_col.items():
            d = frames[sname][frames[sname].model == m].copy()
            if d.empty:
                continue
            bi = d["relmae_vs_snaive"].idxmin()
            wi = d["relmae_vs_snaive"].idxmax()
            b, w = d.loc[bi], d.loc[wi]
            # худший срез на ТОМ ЖЕ горизонте, где модель лучше всего (без тривиального «h=12 хуже»)
            same_h = d[d.horizon == b.horizon]
            wi2 = same_h["relmae_vs_snaive"].idxmax()
            w2 = same_h.loc[wi2]
            txt_win = (f"лучше всего: {b[scol]} (h={int(b.horizon)}, RelMAE={b.relmae_vs_snaive:.3f}, "
                       f"MAE={b.mae_rub:.0f} руб, n={int(b.n_series)})")
            txt_break = (f"хуже всего: {w[scol]} (h={int(w.horizon)}, RelMAE={w.relmae_vs_snaive:.3f}); "
                         f"на горизонте силы h={int(b.horizon)} слабейший срез — {w2[scol]} "
                         f"(RelMAE={w2.relmae_vs_snaive:.3f})")
            rows.append({"model": m, "slice_type": sname,
                         "best_slice": b[scol], "best_h": int(b.horizon),
                         "best_relmae": round(b.relmae_vs_snaive, 4), "best_mae_rub": round(b.mae_rub, 1),
                         "worst_slice": w[scol], "worst_h": int(w.horizon),
                         "worst_relmae": round(w.relmae_vs_snaive, 4),
                         "worst_slice_at_best_h": w2[scol],
                         "worst_relmae_at_best_h": round(w2.relmae_vs_snaive, 4),
                         "kогда_выигрывает": txt_win, "когда_ломается": txt_break})
    return pd.DataFrame(rows)


def load_matched(models_needed: list[str]) -> pd.DataFrame:
    """Matched cells ровно как 04_build_tables.py (пересечение 20 моделей T1 по горизонту),
    но y_pred грузим только для models_needed."""
    key_sets = {}
    keep_parts = []
    files = [f for f in sorted((ROOT / "outputs").glob("preds_*.parquet"))
             if not any(f.match(p) for p in EXCLUDE_GLOB)]
    for f in files:
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        df = df[df["model"] != "y"]
        df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
        for (m, h), g in df.groupby(["model", "horizon"]):
            key_sets.setdefault(h, {})[m] = set(zip(g["series_id"], g["origin"]))
        if df["model"].isin(models_needed).any():
            keep_parts.append(df[df["model"].isin(models_needed)])
    inter = {h: set.intersection(*sets.values()) for h, sets in key_sets.items()}
    df = pd.concat(keep_parts, ignore_index=True)
    df = df[df.apply(lambda r: (r["series_id"], r["origin"]) in inter[r["horizon"]], axis=1)]
    return df


def slice_meta() -> pd.DataFrame:
    """Метаданные разрезов — побитово как в 04_build_tables.py."""
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    train = panel[panel["month_idx"] <= 11]
    vol = train.groupby(["territory_id", "category"])["log_value"].apply(
        lambda s: s.diff().std()).rename("vol").reset_index()
    meta = panel.groupby(["territory_id", "category"]).agg(
        lvl=("value", "median"), region=("region_name", "first"),
        mtype=("municipal_district_type", "first")).reset_index()
    meta["series_id"] = meta["territory_id"].astype(str) + "|" + meta["category"]
    meta = meta.merge(vol, on=["territory_id", "category"], how="left")
    meta["scale_q"] = pd.qcut(meta.groupby("category")["lvl"].rank(method="first"), 5,
                              labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    meta["vol_t"] = pd.qcut(meta.groupby("category")["vol"].rank(method="first"), 3,
                            labels=["V1", "V2", "V3"]).astype(str)
    meta["category"] = meta["series_id"].str.split("|").str[1]
    return meta.set_index("series_id")


def part_b_champion_rule() -> tuple[pd.DataFrame, pd.DataFrame]:
    t1 = pd.read_csv(ROOT / "outputs" / "T1_model_x_horizon.csv")
    t2c = pd.read_csv(ROOT / "outputs" / "T2_slice_category.csv")
    bstar_t1 = t1[t1.model == "Prophet-B*"].set_index("horizon")["mae_rub"]
    bstar_t2 = t2c[(t2c.model == "Prophet-B*") & (t2c.category == "Все категории")].set_index("horizon")["mae_rub"]
    surv = []
    for m in sorted(set(t1.model) - {"Prophet-B*"}):
        mt1 = t1[t1.model == m].set_index("horizon")["mae_rub"]
        mt2 = t2c[(t2c.model == m) & (t2c.category == "Все категории")].set_index("horizon")["mae_rub"]
        ok_pooled = all(mt1[h] < bstar_t1[h] for h in CO_PRIMARY if h in mt1.index)
        ok_head = all(mt2[h] < bstar_t2[h] for h in CO_PRIMARY if h in mt2.index)
        surv.append({"model": m,
                     "ratio_pooled_h1": mt1.get(1, np.nan) / bstar_t1[1],
                     "ratio_pooled_h3": mt1.get(3, np.nan) / bstar_t3[3] if (bstar_t3 := bstar_t1)[3] else np.nan,
                     "ratio_headline_h1": mt2.get(1, np.nan) / bstar_t2[1],
                     "ratio_headline_h3": mt2.get(3, np.nan) / bstar_t2[3],
                     "survives_pooled_frame": bool(ok_pooled and all(h in mt1.index for h in CO_PRIMARY)),
                     "survives_headline_frame": bool(ok_head and all(h in mt2.index for h in CO_PRIMARY))})
    surv = pd.DataFrame(surv)

    candidates = surv[surv.survives_pooled_frame | surv.survives_headline_frame]["model"].tolist()
    for extra in ["DriftAdjSNaive", "SeasProfileHA"]:  # заявленные чемпионы — проверяем в любом случае
        if extra not in candidates:
            candidates.append(extra)
    models_needed = sorted(set(candidates) | {"Prophet-B*"})
    print(f"кандидаты на min-win-rate: {models_needed}", flush=True)

    df = load_matched(models_needed)
    meta = slice_meta()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df["category"] = df["series_id"].str.split("|").str[1]
    for c in ["scale_q", "mtype", "vol_t"]:
        df[c] = df["series_id"].map(meta[c])

    wide = df.pivot_table(index=["series_id", "origin", "horizon", "category", "scale_q", "mtype", "vol_t"],
                          columns="model", values="ae")
    rows = []
    for h in CO_PRIMARY:
        wh = wide[wide.index.get_level_values("horizon") == h]
        for sname, scol in [("category", "category"), ("scale", "scale_q"),
                            ("motype", "mtype"), ("volatility", "vol_t")]:
            for sval, g in wh.groupby(scol, observed=True):
                n_cells = len(g)
                for m in candidates:
                    if m not in g.columns:
                        continue
                    a, b = g[m], g["Prophet-B*"]
                    ok = a.notna() & b.notna()
                    wr_cell = float((a[ok] < b[ok]).mean())
                    # per-series: MAE ряда внутри среза на этом горизонте
                    ps = g.groupby("series_id")[[m, "Prophet-B*"]].mean().dropna()
                    wr_series = float((ps[m] < ps["Prophet-B*"]).mean()) if len(ps) else np.nan
                    rows.append({"horizon": h, "slice_type": sname, "slice": sval, "model": m,
                                 "win_rate_cells": round(wr_cell, 4), "win_rate_series": round(wr_series, 4),
                                 "n_cells": n_cells, "n_series": len(ps)})
    wr = pd.DataFrame(rows)
    # сводка правила: min win rate по всем срезам для каждого кандидата
    rule = (wr.groupby(["horizon", "model"])
              .agg(min_wr_cells=("win_rate_cells", "min"), min_wr_series=("win_rate_series", "min"),
                   argmin_cells=("win_rate_cells", lambda s: s.idxmin()),
                   n_slices=("win_rate_cells", "size")).reset_index())
    rule["argmin_cells"] = rule["argmin_cells"].apply(
        lambda i: f"{wr.loc[i, 'slice_type']}={wr.loc[i, 'slice']}")
    wr.to_csv(ROOT / "outputs" / "T5_champion_rule_by_slice.csv", index=False)
    return surv, rule


def main():
    models = top8_models()
    print("top-8:", models, flush=True)
    cond = part_a_conditions(models)
    cond.to_csv(ROOT / "outputs" / "T5_conditions.csv", index=False)
    print(f"T5_conditions.csv: {len(cond)} строк", flush=True)

    surv, rule = part_b_champion_rule()
    surv.to_csv(ROOT / "outputs" / "T5_champion_rule_screening.csv", index=False)
    rule.to_csv(ROOT / "outputs" / "T5_champion_rule.csv", index=False)
    print("\n=== отбраковка (ratio vs Prophet-B*, co-primary h=1,3) ===")
    print(surv.sort_values("ratio_headline_h1").to_string(index=False), flush=True)
    print("\n=== min win rate по разрезам (правило no_failing_slice) ===")
    print(rule.sort_values(["horizon", "min_wr_cells"], ascending=[True, False]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
