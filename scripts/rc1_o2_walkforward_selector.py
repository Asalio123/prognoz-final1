"""rc1 / O2: честный walk-forward селектор «категория × горизонт → модель».

Вопрос: статическое правило T5 (чемпионы по категориям, T5_category_champions.csv)
выбрано по ПОЛНОЙ сетке origins — описательно, не прогностически. Здесь правило
на origin t фиксируется ТОЛЬКО по прошлому (origin' <= t - h: ошибка ячейки
(origin', h) наблюдаема, когда известен y месяца origin'+h-1 <= t-1), применяется
к origin t. Оценка pooled RelMAE (формула 04_build_tables.py: per-series MAE ->
сумма по рядам / то же для SeasonalNaive, matched cells) против:
  (а) одиночных чемпионов: Prophet-B* (pooled) и headline DriftAdjSNaive h<=6 /
      SeasProfileHA h=12 (на срезе «Все категории»);
  (б) оракула: per-cell min AE по ростеру (верхняя граница селектора);
  (в) статического правила T5 по полной сетке (то, что критикуется).

Два гранулярных варианта селектора:
  A: category × horizon
  B: category × scale_q × horizon
Где истории нет (первый origin каждого горизонта; h=12 целиком) — fallback =
headline-правило (вариант с fallback=Prophet-B* тоже считается).

Проверка геометрии: pooled RelMAE всех моделей обязан совпасть с T1 до 1e-9.

Запуск: .venv/bin/python scripts/rc1_o2_walkforward_selector.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
# ровно как 21_t5_conditions.py (T1 собран до появления сидов LGBM)
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*", "preds_chronos2_quant_o12*",
                "preds_chronos2_zs_raw_o12*", "preds_coldstart*", "preds_lgbm_s*"]
HEADLINE = {1: "DriftAdjSNaive", 3: "DriftAdjSNaive", 6: "DriftAdjSNaive", 12: "SeasProfileHA"}
FALLBACK_POOLED = "Prophet-B*"


def load_matched_full() -> pd.DataFrame:
    """Matched cells ровно как 04_build_tables.py (пересечение по горизонту всех
    моделей вне EXCLUDE_GLOB), но со всеми моделями (нужен оракул)."""
    key_sets: dict[int, dict[str, set]] = {}
    parts = []
    files = [f for f in sorted((ROOT / "outputs").glob("preds_*.parquet"))
             if not any(f.match(p) for p in EXCLUDE_GLOB)]
    for f in files:
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        df = df[df["model"] != "y"]
        df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
        df["origin"] = df["origin"].astype(int)  # в части файлов origin float64 (12.0) — ломало ключи
        df["horizon"] = df["horizon"].astype(int)
        for (m, h), g in df.groupby(["model", "horizon"]):
            key_sets.setdefault(h, {}).setdefault(m, set()).update(zip(g["series_id"], g["origin"]))
        parts.append(df)
    inter = {h: set.intersection(*sets.values()) for h, sets in key_sets.items()}
    df = pd.concat(parts, ignore_index=True)
    keys = df["series_id"] + "|" + df["origin"].astype(str) + "|" + df["horizon"].astype(str)
    inter_flat = {f"{s}|{o}|{h}" for h, ss in inter.items() for s, o in ss}
    return df[keys.isin(inter_flat)].copy()


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    """scale_q — побитово как 04_build_tables.py / 21_t5_conditions.py."""
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    meta = panel.groupby(["territory_id", "category"]).agg(
        lvl=("value", "median")).reset_index()
    meta["series_id"] = meta["territory_id"].astype(str) + "|" + meta["category"]
    meta["scale_q"] = pd.qcut(meta.groupby("category")["lvl"].rank(method="first"), 5,
                              labels=["Q1", "Q2", "Q3", "Q4", "Q5"]).astype(str)
    df["category"] = df["series_id"].str.split("|").str[1]
    df["scale_q"] = df["series_id"].map(meta.set_index("series_id")["scale_q"])
    return df


def pooled_relmae(ae, ae_sn, series) -> float:
    """Формула 04: per-series mean AE, сумма по рядам / то же SNaive (те же ячейки)."""
    d = pd.DataFrame({"ae": np.asarray(ae), "ae_sn": np.asarray(ae_sn),
                      "series_id": np.asarray(series)})
    ps = d.groupby("series_id")[["ae", "ae_sn"]].mean()
    return float(ps["ae"].sum() / ps["ae_sn"].sum())


def verify_t1(df: pd.DataFrame) -> None:
    t1 = pd.read_csv(ROOT / "outputs" / "T1_model_x_horizon.csv")
    sn = df[df.model == "SeasonalNaive"].set_index(["series_id", "origin", "horizon"])["ae"]
    max_diff = 0.0
    for (m, h), g in df.groupby(["model", "horizon"]):
        ref = t1[(t1.model == m) & (t1.horizon == h)]
        if ref.empty:
            continue
        idx = list(zip(g["series_id"], g["origin"], g["horizon"]))
        ae_sn = sn.reindex(idx).to_numpy()
        rel = pooled_relmae(g["ae"], pd.Series(ae_sn, index=g.index), g["series_id"])
        max_diff = max(max_diff, abs(rel - ref["relmae_vs_snaive"].iloc[0]))
    print(f"[check] max |RelMAE - T1| = {max_diff:.2e}", flush=True)
    assert max_diff < 1e-9, "геометрия matched cells НЕ совпадает с T1"


def build_rules(dfh: pd.DataFrame, group_cols: list[str], fallback: dict[int, str],
                agg: str = "sum"):
    """Для каждого origin t и группы: argmin AE по ячейкам с origin' <= t - h.
    agg="sum": сумма AE (pooled-метрика, вес ~ масштабу ряда);
    agg="series_mean": mean per-series AE (равный вес ряда, как per-series шаг 04).
    Возвращает DataFrame (group_cols, origin, model) и выбранные AE на ячейках."""
    h = int(dfh["horizon"].iloc[0])
    origins = sorted(dfh["origin"].unique())
    wide = dfh.pivot_table(index=["series_id", "origin"] + group_cols,
                           columns="model", values="ae")
    wide = wide.sort_index()
    rules, chosen = [], []
    for t in origins:
        hist = wide[wide.index.get_level_values("origin") <= t - h]
        cur = wide[wide.index.get_level_values("origin") == t]
        for gkey, gc in cur.groupby(group_cols, observed=True):
            best = fallback[h]
            if len(hist):
                mask = np.ones(len(hist), dtype=bool)
                for col, val in zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,)):
                    mask &= (hist.index.get_level_values(col) == val)
                gh = hist[mask]
                if agg == "sum":
                    scores = gh.sum()
                else:  # series_mean: сначала mean по origins внутри ряда, потом по рядам
                    scores = gh.groupby(level="series_id").mean().mean()
                scores = scores.dropna()
                if len(scores):
                    best = scores.idxmin()  # ties -> первый по алфавиту (idxmin стабилен)
            rules.append({**dict(zip(group_cols, gkey if isinstance(gkey, tuple) else (gkey,))),
                          "origin": t, "model": best, "had_history": bool(len(hist))})
            aec = gc[best] if best in gc.columns else gc[fallback[h]]
            chosen.append(aec)
    rules = pd.DataFrame(rules)
    ae_sel = pd.concat(chosen).rename("ae_sel")
    return rules, ae_sel


def main():
    print("load matched...", flush=True)
    df = load_matched_full()
    df["ae"] = (df["y_true"] - df["y_pred"]).abs()
    df = add_meta(df)
    print(f"  {len(df)} строк, {df.model.nunique()} моделей", flush=True)
    verify_t1(df)

    sn = df[df.model == "SeasonalNaive"].set_index(["series_id", "origin", "horizon"])["ae"]
    idx = list(zip(df["series_id"], df["origin"], df["horizon"]))
    df["ae_sn"] = sn.reindex(idx).to_numpy()

    static = pd.read_csv(ROOT / "outputs" / "T5_category_champions.csv")
    static_map = {(r.category, int(r.horizon)): r.champion for r in static.itertuples()}

    wide_all = {h: g.pivot_table(index=["series_id", "origin", "category", "scale_q"],
                                 columns="model", values="ae")
                for h, g in df.groupby("horizon")}
    sn_by_h = {h: df[df.horizon == h].drop_duplicates(["series_id", "origin"])
               .set_index(["series_id", "origin"])["ae_sn"] for h in HORIZONS}

    models_by_h = {h: sorted(w.columns) for h, w in wide_all.items()}
    oracle = {}
    for h, w in wide_all.items():
        oracle[h] = w.min(axis=1)

    # --- правила селекторов: A/B = argmin ΣAE, C/D = argmin mean per-series AE ---
    rules_frames, sel_ae = {}, {}
    for h in HORIZONS:
        dfh = df[df.horizon == h]
        ra, sa = build_rules(dfh, ["category"], HEADLINE, agg="sum")
        rb, sb = build_rules(dfh, ["category", "scale_q"], HEADLINE, agg="sum")
        rc, sc = build_rules(dfh, ["category"], HEADLINE, agg="series_mean")
        rd, sd = build_rules(dfh, ["category", "scale_q"], HEADLINE, agg="series_mean")
        for name, r_, s_ in [("A", ra, sa), ("B", rb, sb), ("C", rc, sc), ("D", rd, sd)]:
            rules_frames[(h, name)] = r_
            sel_ae[(h, name)] = s_

    rules_all = pd.concat([r.assign(horizon=h, granularity=g)
                           for (h, g), r in rules_frames.items()], ignore_index=True)
    rules_all.to_csv(ROOT / "outputs" / "rc1_o2_selector_rules.csv", index=False)

    # --- оценка: пул вариантов (всё в индексном пространстве wide-таблиц) ---
    def eval_variants(only_category=None, origins_with_hist_only=False):
        rows = []
        for h in HORIZONS:
            w = wide_all[h]
            idx = w.index
            if origins_with_hist_only:
                keep = [t for t in idx.get_level_values("origin").unique() if t - h >= 12]
                idx = idx[idx.get_level_values("origin").isin(keep)]
            if only_category is not None:
                idx = idx[idx.get_level_values("category") == only_category]
            if len(idx) == 0:
                continue
            wl = w.loc[idx]
            series = idx.get_level_values("series_id")
            ae_sn = sn_by_h[h].reindex(
                list(zip(series, idx.get_level_values("origin")))).to_numpy()

            def put(name, ae_values):
                rows.append({"horizon": h, "variant": name,
                             "relmae": pooled_relmae(pd.Series(ae_values, index=idx),
                                                     pd.Series(ae_sn, index=idx),
                                                     pd.Series(series, index=idx)),
                             "n_origins": int(idx.get_level_values("origin").nunique())})

            put("oracle", oracle[h].loc[idx].to_numpy())
            put("Prophet-B*", wl["Prophet-B*"].to_numpy())
            put("headline_" + HEADLINE[h], wl[HEADLINE[h]].to_numpy())
            # статическое правило полной сетки: (category, h) -> чемпион T5
            cats = idx.get_level_values("category")
            static_models = [static_map.get((c, h), "Prophet-B*") for c in cats]
            cols = {m: i for i, m in enumerate(w.columns)}
            ae_static = wl.to_numpy()[np.arange(len(wl)), [cols[m] for m in static_models]]
            put("static_T5_rule", ae_static)
            for gname in ["A", "B", "C", "D"]:
                sa = sel_ae[(h, gname)]
                if gname in ("A", "C"):  # индекс без scale_q
                    key = list(zip(series, idx.get_level_values("origin"), cats))
                else:
                    key = list(idx)
                put(f"selector_{gname}", sa.loc[key].to_numpy())
        return pd.DataFrame(rows)

    print("evaluate...", flush=True)
    res_all = eval_variants()
    res_hist = eval_variants(origins_with_hist_only=True)
    res_vse = eval_variants(only_category="Все категории")
    res_all.to_csv(ROOT / "outputs" / "rc1_o2_selector_summary.csv", index=False)
    res_hist.to_csv(ROOT / "outputs" / "rc1_o2_selector_summary_histonly.csv", index=False)
    res_vse.to_csv(ROOT / "outputs" / "rc1_o2_selector_summary_vsekategorii.csv", index=False)

    # --- gap analysis ---
    def gap_table(res, champ_name_fn, tag):
        rows = []
        for h in res["horizon"].unique():
            r = res[res.horizon == h].set_index("variant")["relmae"]
            for sel in ["selector_A", "selector_B", "selector_C", "selector_D", "static_T5_rule"]:
                champ = champ_name_fn(h)
                if champ not in r.index or sel not in r.index or "oracle" not in r.index:
                    continue
                gap = r[champ] - r["oracle"]
                closed = (r[champ] - r[sel]) / gap if gap > 0 else np.nan
                rows.append({"frame": tag, "horizon": h, "vs": champ, "selector": sel,
                             "champ_relmae": round(r[champ], 4), "sel_relmae": round(r[sel], 4),
                             "oracle_relmae": round(r["oracle"], 4),
                             "gap_closed_pct": round(100 * closed, 1)})
        return pd.DataFrame(rows)

    g1 = gap_table(res_all, lambda h: "Prophet-B*", "pooled")
    g2 = gap_table(res_vse, lambda h: "headline_" + HEADLINE[h], "vsekategorii")
    g3 = gap_table(res_hist, lambda h: "Prophet-B*", "pooled_histonly")
    gap = pd.concat([g1, g2, g3], ignore_index=True)
    gap.to_csv(ROOT / "outputs" / "rc1_o2_gap_closed.csv", index=False)

    # --- Маркетплейсы и стабильность ---
    mp = rules_all[(rules_all.category == "Маркетплейсы") & (rules_all.granularity == "A")]
    mp_piv = mp.pivot_table(index="origin", columns="horizon", values="model", aggfunc="first")
    print("\n=== Маркетплейсы: выбранная модель по origin (селектор A) ===\n", mp_piv.to_string(), flush=True)

    stab = []
    for (h, gname, cat), r in rules_all.groupby(["horizon", "granularity", "category"]):
        r = r.sort_values("origin")
        r_h = r[r.had_history]
        if len(r_h) < 2:
            continue
        same = (r_h["model"].to_numpy()[1:] == r_h["model"].to_numpy()[:-1]).mean()
        last = r_h.iloc[-1]
        stab.append({"horizon": h, "granularity": gname, "category": cat,
                     "n_origins_with_hist": len(r_h), "n_distinct_models": r_h["model"].nunique(),
                     "share_same_as_prev": round(float(same), 3),
                     "last_origin_model": last["model"],
                     "static_rule_model": static_map.get((cat, h), "n/a"),
                     "last_matches_static": bool(last["model"] == static_map.get((cat, h)))})
    stab = pd.DataFrame(stab)
    stab.to_csv(ROOT / "outputs" / "rc1_o2_rule_stability.csv", index=False)

    # --- per-category деталь: селектор vs статика vs headline vs чемпион T5 ---
    det = []
    for h in HORIZONS:
        w = wide_all[h]
        idx = w.index
        series = idx.get_level_values("series_id")
        cats = idx.get_level_values("category")
        ae_sn = sn_by_h[h].reindex(list(zip(series, idx.get_level_values("origin")))).to_numpy()
        keyA = list(zip(series, idx.get_level_values("origin"), cats))
        for c in sorted(set(cats)):
            m = cats == c
            for name, vals in [("selector_A", sel_ae[(h, "A")].loc[keyA].to_numpy()),
                               ("headline", w[HEADLINE[h]].to_numpy()),
                               ("Prophet-B*", w["Prophet-B*"].to_numpy()),
                               ("T5_champion_fullgrid", w[static_map.get((c, h), "Prophet-B*")].to_numpy())]:
                det.append({"horizon": h, "category": c, "variant": name,
                            "relmae": pooled_relmae(vals[m], ae_sn[m], series[m])})
    det = pd.DataFrame(det)
    det.to_csv(ROOT / "outputs" / "rc1_o2_selector_by_category.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== SUMMARY pooled (все origins, fallback=headline) ===\n", res_all.to_string(index=False), flush=True)
    print("\n=== SUMMARY pooled (только origins с историей) ===\n", res_hist.to_string(index=False), flush=True)
    print("\n=== SUMMARY «Все категории» ===\n", res_vse.to_string(index=False), flush=True)
    print("\n=== GAP CLOSED ===\n", gap.to_string(index=False), flush=True)
    print("\n=== Стабильность правила (по категориям, селектор A) ===\n",
          stab[stab.granularity == "A"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
