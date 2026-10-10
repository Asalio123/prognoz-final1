"""Майнинг энсемблей из существующих прогнозов (К2).

Протокол (зафиксирован ДО запуска):
- Данные и matched cells: дословно как scripts/04_build_tables.py (те же EXCLUDE_GLOB,
  dedupe keep=last, пересечение series×origin по ВСЕМ моделям внутри горизонта).
- Сплит по origins (selection / holdout), на горизонте:
    h=1: origins 12..23 -> sel 12..19 (8), holdout 20..23 (4)
    h=3: origins 12..21 -> sel 12..18 (7), holdout 19..21 (3)
    h=6: origins 12..18 -> sel 12..16 (5), holdout 17..18 (2)
    h=12: 1 origin -> честного сплита нет, ТОЛЬКО диагностика full-sample
- Пул членов энсембля: все модели T1, КРОМЕ SeasonalNaive (референс RelMAE),
  Ensemble-median3 (существующий энсембль) и сид-реплик LightGBM-direct_s*
  (диагностика стабильности сидов, не независимые модели).
- Кандидаты (для каждого среза pooled / «Все категории» и каждого горизонта):
    1. mean-top3, 2. median-top3, 3. mean-top5, 4. median-top5
       (top-k по RelMAE на SELECTION origins того же среза)
    5. invweighted-top5 (веса = 1/RelMAE на selection)
    6. trimmed-mean-top5 (на ячейке отбрасываем min/max члена, среднее остальных 3)
    7. fixed 50/50 DriftAdjSNaive + Prophet-B* (состав априорный, без селекции)
  Для h=12: только кандидат 7 (fixed) + перенос составов top-k с h=6 (помечено
  "transfer_h6"), full-sample, без гейта.
- Оценка: ТОЛЬКО на holdout origins. RelMAE = sum per-series MAE / sum per-series
  MAE SNaive на тех же ячейках; MAE_rub = mean per-series MAE (как в 04).
- Гейт (зеркало замороженного из 07_run_ensemble.py): ΔRelMAE <= -1% против
  лучшей ОДИНОЧНОЙ модели на holdout (лучшая одиночная = min holdout RelMAE среди
  всех не-энсемблей). Co-primary: h=1, h=3. h=6 — secondary. h=12 — без гейта.
- MCS (бонус): h=1 pooled, HLN T_max-вариант, block bootstrap по регионам,
  B=1000, alpha=0.10, seed=20261007.

Выход: outputs/rc1_o1_candidates.csv, rc1_o1_holdout_eval.csv,
       rc1_o1_fullsample_eval.csv, rc1_o1_mcs_h1.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*",
                "preds_chronos2_quant_o12*", "preds_chronos2_zs_raw_o12*",
                "preds_coldstart*"]
POOL_EXCLUDE = {"SeasonalNaive", "Ensemble-median3",
                "LightGBM-direct_s137", "LightGBM-direct_s1618",
                "LightGBM-direct_s271", "LightGBM-direct_s314"}
ENSEMBLE_NAMES_EXISTING = {"Ensemble-median3"}
SPLIT = {1: (list(range(12, 20)), list(range(20, 24))),
         3: (list(range(12, 19)), list(range(19, 22))),
         6: (list(range(12, 17)), list(range(17, 19)))}
HORIZONS = [1, 3, 6, 12]


def load_all() -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_*.parquet")):
        if any(f.match(p) for p in EXCLUDE_GLOB):
            continue
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["model"] != "y"]
    df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
    df["series_id"] = df["series_id"].astype("category")
    df["model"] = df["model"].astype("category")
    df["horizon"] = df["horizon"].astype(int)
    df["origin"] = df["origin"].astype(int)
    return df


def matched_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Пересечение (series_id, origin) по всем моделям внутри горизонта (как 04)."""
    keep = []
    for h, g in df.groupby("horizon", observed=True):
        sid = g["series_id"].astype(str)
        inter = None
        for m, gg in g.groupby("model", observed=True):
            s = set(zip(gg["series_id"].astype(str), gg["origin"]))
            inter = s if inter is None else inter & s
        key = pd.MultiIndex.from_arrays([sid, g["origin"]])
        mask = key.isin(inter)
        keep.append(g[mask])
    return pd.concat(keep, ignore_index=True)


def add_snaive(df: pd.DataFrame) -> pd.DataFrame:
    sn = df[df["model"] == "SeasonalNaive"][["series_id", "origin", "horizon", "y_pred"]]
    sn = sn.rename(columns={"y_pred": "y_snaive"})
    return df.merge(sn, on=["series_id", "origin", "horizon"], how="left")


def add_category(df: pd.DataFrame) -> pd.DataFrame:
    cat = pd.Series(df["series_id"].astype(str).str.split("|").str[1],
                    index=df.index, name="category")
    return df.assign(category=cat.values)


def eval_metrics(d: pd.DataFrame) -> dict:
    """MAE_rub и RelMAE как в 04: per-series mean AE -> mean / sum-ratio."""
    d = d.copy()
    d["ae"] = (d["y_true"] - d["y_pred"]).abs()
    d["ae_sn"] = (d["y_true"] - d["y_snaive"]).abs()
    ps = d.groupby("series_id", observed=True).agg(ae=("ae", "mean"), ae_sn=("ae_sn", "mean"))
    return {"mae_rub": ps["ae"].mean(),
            "relmae": ps["ae"].sum() / ps["ae_sn"].sum(),
            "n_series": int(ps.shape[0]),
            "n_origins": int(d["origin"].nunique())}


def build_ensemble_pred(member_df: pd.DataFrame, members: list[str], how: str,
                        weights: pd.Series | None = None) -> pd.DataFrame:
    """Возвращает y_pred энсембля на ячейках member_df (только members)."""
    wide = (member_df[member_df["model"].isin(members)]
            .pivot_table(index=["series_id", "origin", "horizon"], columns="model",
                         values="y_pred", observed=True))
    wide = wide[members]  # фиксированный порядок
    if how == "mean":
        pred = wide.mean(axis=1)
    elif how == "median":
        pred = wide.median(axis=1)
    elif how == "invw":
        w = np.array([weights[m] for m in members], dtype=float)
        pred = wide.to_numpy() @ w / w.sum()
        pred = pd.Series(pred, index=wide.index)
    elif how == "trimmed":
        a = np.sort(wide.to_numpy(), axis=1)
        pred = pd.Series(a[:, 1:-1].mean(axis=1), index=wide.index)
    else:
        raise ValueError(how)
    out = pred.rename("y_pred").reset_index()
    yt = (member_df.drop_duplicates(["series_id", "origin", "horizon"])
          .set_index(["series_id", "origin", "horizon"])["y_true"])
    out["y_true"] = out.set_index(["series_id", "origin", "horizon"]).index.map(yt)
    return out


def main():
    t0 = time.time()
    print("[1/6] load...", flush=True)
    df = load_all()
    print(f"    {len(df)} строк, {df['model'].nunique()} моделей", flush=True)

    print("[2/6] matched cells + snaive + category...", flush=True)
    dm = matched_cells(df)
    del df
    dm = add_snaive(dm)
    dm = add_category(dm)
    print(f"    matched: {len(dm)} строк", flush=True)

    # ---- верификация против T1 ----
    print("[3/6] верификация против outputs/T1_model_x_horizon.csv...", flush=True)
    t1 = pd.read_csv(ROOT / "outputs" / "T1_model_x_horizon.csv")
    check_rows = []
    for (m, h), g in dm.groupby(["model", "horizon"], observed=True):
        met = eval_metrics(g)
        ref = t1[(t1["model"] == m) & (t1["horizon"] == h)]
        if len(ref):
            check_rows.append({"model": m, "horizon": h,
                               "relmae_mine": met["relmae"],
                               "relmae_T1": ref["relmae_vs_snaive"].iloc[0],
                               "mae_mine": met["mae_rub"],
                               "mae_T1": ref["mae_rub"].iloc[0]})
    chk = pd.DataFrame(check_rows)
    chk["d_relmae"] = (chk["relmae_mine"] - chk["relmae_T1"]).abs()
    chk["d_mae"] = (chk["mae_mine"] - chk["mae_T1"]).abs()
    print(f"    max |dRelMAE| = {chk['d_relmae'].max():.3e}, max |dMAE| = {chk['d_mae'].max():.3e} "
          f"на {len(chk)} строках model×horizon", flush=True)
    chk.to_csv(ROOT / "outputs" / "rc1_o1_t1_verification.csv", index=False)

    # ---- майнинг ----
    print("[4/6] кандидаты и holdout-оценка...", flush=True)
    cand_rows, eval_rows, full_rows = [], [], []
    singles = sorted(set(dm["model"].astype(str).unique()) - ENSEMBLE_NAMES_EXISTING)
    pool = sorted(set(singles) - POOL_EXCLUDE)

    for slice_name, dslice in [("pooled", dm),
                               ("Все категории", dm[dm["category"] == "Все категории"])]:
        for h in HORIZONS:
            dh = dslice[dslice["horizon"] == h]
            if len(dh) == 0:
                continue
            sn_ref = dh[dh["model"] == "SeasonalNaive"][
                ["series_id", "origin", "horizon", "y_snaive"]].drop_duplicates()

            # --- одиночные: метрики sel/holdout/full ---
            single_sel, single_hold, single_full = {}, {}, {}
            for m in singles:
                gm = dh[dh["model"] == m]
                if len(gm) == 0:
                    continue
                single_full[m] = eval_metrics(gm)
                if h in SPLIT:
                    sel_o, hold_o = SPLIT[h]
                    single_sel[m] = eval_metrics(gm[gm["origin"].isin(sel_o)])
                    single_hold[m] = eval_metrics(gm[gm["origin"].isin(hold_o)])

            # --- составы кандидатов (ТОЛЬКО selection origins) ---
            candidates = []  # (name, members, how, weights, transfer_tag)
            if h in SPLIT:
                ranking = pd.Series({m: v["relmae"] for m, v in single_sel.items()
                                     if m in pool}).sort_values()
                top3 = list(ranking.index[:3])
                top5 = list(ranking.index[:5])
                invw = {m: 1.0 / single_sel[m]["relmae"] for m in top5}
                candidates += [
                    ("mean-top3", top3, "mean", None, ""),
                    ("median-top3", top3, "median", None, ""),
                    ("mean-top5", top5, "mean", None, ""),
                    ("median-top5", top5, "median", None, ""),
                    ("invw-top5", top5, "invw", invw, ""),
                    ("trimmed-top5", top5, "trimmed", None, ""),
                    ("fixed-50/50-DriftAdj+ProphetB*", ["DriftAdjSNaive", "Prophet-B*"],
                     "mean", None, ""),
                ]
                for name, members, how, w, tag in candidates:
                    cand_rows.append({"slice": slice_name, "horizon": h, "candidate": name,
                                      "members": "|".join(members), "transfer": tag,
                                      "sel_origins": SPLIT[h][0], "hold_origins": SPLIT[h][1]})
            else:  # h=12: честного сплита нет
                # fixed 50/50 + перенос составов с h=6 (та же срез)
                h6 = dslice[dslice["horizon"] == 6]
                sel6, _ = SPLIT[6]
                rank6 = {}
                for m in pool:
                    gm = h6[(h6["model"] == m) & (h6["origin"].isin(sel6))]
                    if len(gm):
                        rank6[m] = eval_metrics(gm)["relmae"]
                rank6 = pd.Series(rank6).sort_values()
                top3_6, top5_6 = list(rank6.index[:3]), list(rank6.index[:5])
                candidates = [
                    ("fixed-50/50-DriftAdj+ProphetB*", ["DriftAdjSNaive", "Prophet-B*"],
                     "mean", None, ""),
                    ("mean-top3", top3_6, "mean", None, "transfer_h6sel"),
                    ("median-top3", top3_6, "median", None, "transfer_h6sel"),
                    ("mean-top5", top5_6, "mean", None, "transfer_h6sel"),
                    ("median-top5", top5_6, "median", None, "transfer_h6sel"),
                ]
                for name, members, how, w, tag in candidates:
                    cand_rows.append({"slice": slice_name, "horizon": h, "candidate": name,
                                      "members": "|".join(members), "transfer": tag,
                                      "sel_origins": None, "hold_origins": None})

            # --- оценка кандидатов ---
            for name, members, how, w, tag in candidates:
                avail = [m for m in members if m in set(dh["model"])]
                if len(avail) < len(members):
                    continue
                ens = build_ensemble_pred(dh, members, how, w)
                ens = ens.merge(sn_ref, on=["series_id", "origin", "horizon"], how="left")
                assert ens["y_snaive"].notna().all()
                full_met = eval_metrics(ens)
                row = {"slice": slice_name, "horizon": h, "candidate": name,
                       "transfer": tag, **full_met}
                full_rows.append(row)
                if h in SPLIT:
                    sel_o, hold_o = SPLIT[h]
                    hold_met = eval_metrics(ens[ens["origin"].isin(hold_o)])
                    sel_met = eval_metrics(ens[ens["origin"].isin(sel_o)])
                    best_single_hold = min(single_hold.items(), key=lambda kv: kv[1]["relmae"])
                    d_pct = (hold_met["relmae"] / best_single_hold[1]["relmae"] - 1) * 100
                    eval_rows.append({"slice": slice_name, "horizon": h, "candidate": name,
                                      "sel_relmae": sel_met["relmae"],
                                      "hold_relmae": hold_met["relmae"],
                                      "hold_mae_rub": hold_met["mae_rub"],
                                      "best_single_holdout": best_single_hold[0],
                                      "best_single_holdout_relmae": best_single_hold[1]["relmae"],
                                      "delta_pct_vs_best_single": d_pct,
                                      "gate_pass": bool(d_pct <= -1.0)})

    cand_df = pd.DataFrame(cand_rows)
    eval_df = pd.DataFrame(eval_rows)
    full_df = pd.DataFrame(full_rows)
    cand_df.to_csv(ROOT / "outputs" / "rc1_o1_candidates.csv", index=False)
    eval_df.to_csv(ROOT / "outputs" / "rc1_o1_holdout_eval.csv", index=False)
    full_df.to_csv(ROOT / "outputs" / "rc1_o1_fullsample_eval.csv", index=False)
    print(eval_df.sort_values(["slice", "horizon", "hold_relmae"]).to_string(index=False), flush=True)

    # ---- MCS (h=1, pooled, block bootstrap по регионам) ----
    print("[5/6] MCS h=1 (block bootstrap по регионам, B=1000)...", flush=True)
    d1 = dm[dm["horizon"] == 1]
    wide = d1.pivot_table(index=["series_id", "origin"], columns="model", values="y_pred",
                          observed=True)
    yt = d1.drop_duplicates(["series_id", "origin"]).set_index(["series_id", "origin"])["y_true"]
    loss = wide.sub(yt, axis=0).abs().to_numpy(dtype=np.float64)  # cells × models
    models = list(wide.columns)
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet",
                            columns=["territory_id", "category", "region_name"])
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    reg_map = panel.drop_duplicates("series_id").set_index("series_id")["region_name"]
    cell_series = wide.index.get_level_values("series_id").astype(str)
    regions = cell_series.map(reg_map)
    n_na = int(regions.isna().sum())
    if n_na:
        print(f"    WARNING: {n_na} ячеек без региона — исключены из MCS", flush=True)
        ok = ~regions.isna()
        loss, regions = loss[ok], regions[ok]
    reg_codes, reg_uniq = pd.factorize(regions)
    R = len(reg_uniq)
    # суммы потерь по регионам: S (R × M), счётчики n (R)
    S = np.zeros((R, loss.shape[1]))
    np.add.at(S, reg_codes, loss)
    n = np.bincount(reg_codes).astype(float)
    rng = np.random.default_rng(20261007)
    B, alpha = 1000, 0.10
    W = np.zeros((B, R))
    for b in range(B):
        W[b] = np.bincount(rng.integers(0, R, R), minlength=R)
    denom = W @ n  # (B,)

    active = list(range(len(models)))
    history = []
    while True:
        idx = np.array(active)
        Lbar = (S[:, idx].sum(axis=0)) / n.sum()  # наблюдаемые mean loss
        dbar = Lbar - Lbar.mean()
        T_obs = np.abs(dbar).max()
        Lb = (W @ S[:, idx]) / denom[:, None]     # (B × |M|)
        db = Lb - Lb.mean(axis=1, keepdims=True)
        Tb = np.abs(db - dbar).max(axis=1)
        p = float((Tb >= T_obs).mean())
        worst = int(idx[np.argmax(dbar)])
        history.append({"step": len(history), "n_models": len(active),
                        "T_obs": T_obs, "p_value": p,
                        "eliminated": models[worst] if p < alpha else ""})
        if p >= alpha or len(active) <= 2:
            break
        active.remove(worst)
    mcs = pd.DataFrame({"model": [models[i] for i in active], "in_MCS90": True})
    allm = pd.DataFrame({"model": models})
    mcs = allm.merge(mcs, on="model", how="left").fillna({"in_MCS90": False})
    mcs.to_csv(ROOT / "outputs" / "rc1_o1_mcs_h1.csv", index=False)
    hist = pd.DataFrame(history)
    hist.to_csv(ROOT / "outputs" / "rc1_o1_mcs_h1_steps.csv", index=False)
    print(f"    MCS90 ({len(active)} моделей): {[models[i] for i in active]}", flush=True)
    print(hist.to_string(index=False), flush=True)

    print(f"[6/6] done за {time.time() - t0:.0f} с", flush=True)


if __name__ == "__main__":
    main()
