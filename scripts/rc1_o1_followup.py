"""rc1 / O1 follow-up: независимый пересчёт singles на selection/holdout origins.

Второй метод относительно rc1_o1_ensemble_mining.py: matched-пересечение там
ничего не отбросило (9006822 строк до и после), значит per-model ячейки уже
идентичны — считаю напрямую из файлов без matched-машинерии. Также кросс-чек
двух ключевых holdout-ячеек 50/50 (pooled h=1 и h=3).

Выход: outputs/rc1_o1_singles_splits.csv, outputs/rc1_o1_crosscheck.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SPLIT = {1: (list(range(12, 20)), list(range(20, 24))),
         3: (list(range(12, 19)), list(range(19, 22))),
         6: (list(range(12, 17)), list(range(17, 19)))}
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*",
                "preds_chronos2_quant_o12*", "preds_chronos2_zs_raw_o12*",
                "preds_coldstart*", "preds_ensemble*"]


def load_all():
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
    return df


def relmae(d):
    d = d.assign(ae=(d["y_true"] - d["y_pred"]).abs(),
                 ae_sn=(d["y_true"] - d["y_snaive"]).abs())
    ps = d.groupby("series_id").agg(ae=("ae", "mean"), ae_sn=("ae_sn", "mean"))
    return ps["ae"].sum() / ps["ae_sn"].sum(), ps["ae"].mean()


def main():
    df = load_all()
    sn = (df[df["model"] == "SeasonalNaive"][["series_id", "origin", "horizon", "y_pred"]]
          .rename(columns={"y_pred": "y_snaive"}))
    df = df.merge(sn, on=["series_id", "origin", "horizon"], how="left")
    df["category"] = df["series_id"].str.split("|").str[1]

    t1_models = set(pd.read_csv(ROOT / "outputs" / "T1_model_x_horizon.csv")["model"])
    rows = []
    for slice_name, dslice in [("pooled", df),
                               ("Все категории", df[df["category"] == "Все категории"])]:
        for h, (sel_o, hold_o) in SPLIT.items():
            dh = dslice[dslice["horizon"] == h]
            for m, g in dh.groupby("model"):
                if m == "SeasonalNaive":
                    continue
                r_sel = relmae(g[g["origin"].isin(sel_o)])
                r_hold = relmae(g[g["origin"].isin(hold_o)])
                r_full = relmae(g)
                rows.append({"slice": slice_name, "horizon": h, "model": m,
                             "in_T1": m in t1_models,
                             "sel_relmae": r_sel[0], "hold_relmae": r_hold[0],
                             "full_relmae": r_full[0], "hold_mae_rub": r_hold[1]})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs" / "rc1_o1_singles_splits.csv", index=False)

    # сводка: лучший одиночный на holdout (весь ростер / только T1-ростер)
    for (s, h), g in out.groupby(["slice", "horizon"]):
        b_all = g.loc[g["hold_relmae"].idxmin()]
        b_t1 = g[g["in_T1"]].loc[g[g["in_T1"]]["hold_relmae"].idxmin()]
        print(f"{s:15s} h={h:2d} | best holdout (все): {b_all['model']:28s} {b_all['hold_relmae']:.4f}"
              f" | best holdout (T1): {b_t1['model']:28s} {b_t1['hold_relmae']:.4f}", flush=True)

    # ключевые модели для нарратива
    key = out[out["model"].isin(["Prophet-B*", "DriftAdjSNaive", "SeasProfileHA",
                                 "LightGBM-direct", "LightGBM-direct-news",
                                 "LightGBM-direct_s137", "LightGBM-direct_s271"])]
    print(key.sort_values(["slice", "horizon", "hold_relmae"]).to_string(index=False), flush=True)

    # кросс-чек 50/50 (независимый код): pooled h=1 и h=3, holdout
    cc = []
    for h, (sel_o, hold_o) in [(1, SPLIT[1]), (3, SPLIT[3])]:
        dh = df[(df["horizon"] == h) & df["model"].isin(["DriftAdjSNaive", "Prophet-B*"])]
        wide = dh.pivot_table(index=["series_id", "origin"], columns="model", values="y_pred")
        wide["ens"] = wide.mean(axis=1)
        aux = df[df["horizon"] == h].drop_duplicates(["series_id", "origin"]).set_index(
            ["series_id", "origin"])[["y_true", "y_snaive"]]
        wide = wide.join(aux)
        for split, oo in [("selection", sel_o), ("holdout", hold_o)]:
            w = wide[wide.index.get_level_values("origin").isin(oo)]
            d = w.reset_index()[["series_id", "y_true", "ens", "y_snaive"]].rename(
                columns={"ens": "y_pred"})
            r, mae = relmae(d)
            cc.append({"check": f"pooled h={h} 50/50 {split}", "relmae": r, "mae_rub": mae})
    ccd = pd.DataFrame(cc)
    ccd.to_csv(ROOT / "outputs" / "rc1_o1_crosscheck.csv", index=False)
    print(ccd.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
