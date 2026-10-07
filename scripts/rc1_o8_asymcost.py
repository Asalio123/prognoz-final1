"""rc1 / O8: асимметричная linlin (pinball) стоимость ошибки прогноза (К6/К7).

Мотивация: MAE симметрична, но для ритейла/логистики недопрогноз (пустая полка)
дороже перепрогноза (замороженный оборотный капитал на складе). Linlin-потеря:
    L_k(e) = k * e,  если e = y_true - y_pred > 0  (недопрогноз)
             |e|,     если e <= 0                  (перепрогноз)
Это ровно pinball-потеря квантиля tau = k/(1+k): оптимальный точечный прогноз
под L_k — tau-квантиль условного распределения, а не медиана.

Что считается (всё из готовых outputs/preds_*.parquet, модели НЕ перегоняются):
1. Таблица «модель x k -> средняя linlin-потеря (руб/мес)» на matched cells,
   h in {1,3,6}, модели: DriftAdjSNaive, Prophet-B*, HistoricAverage, SeasonalNaive.
   Pooled (все 11682 ряда) и разрез «Все категории» (1947 МО).
2. Граница k*: при какой асимметрии pooled-чемпион меняется (сетка k с шагом 0.25).
3. Калиброванное смещение: прогноз чемпиона + эмпирический tau=2/3-квантиль
   остатков, калиброванный СТРОГО на прошлых origins (expanding, min 3 origins),
   оценка на будущих origins против точечного прогноза при k=2.

Запуск: .venv/bin/python scripts/rc1_o8_asymcost.py
Выходы: outputs/rc1_o8_linlin.csv, outputs/rc1_o8_kstar.csv, outputs/rc1_o8_shift.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.significance import paired_wilcoxon  # noqa: E402

HORIZONS = [1, 3, 6]  # h=12: 1 origin = лотерея (HANDOFF §4), не используем
MODELS = ["DriftAdjSNaive", "Prophet-B*", "HistoricAverage", "SeasonalNaive"]
K_TABLE = [1.0, 1.5, 2.0, 3.0]
K_SWEEP = np.round(np.arange(1.0, 10.01, 0.25), 2)


def linlin(e: np.ndarray, k: float) -> np.ndarray:
    return np.where(e > 0, k * e, -e)


def load_models() -> pd.DataFrame:
    parts = []
    for f in sorted((ROOT / "outputs").glob("preds_stats_part*.parquet")):
        d = pd.read_parquet(f)
        parts.append(d[d["model"].isin(["HistoricAverage", "SeasonalNaive"])])
    parts.append(pd.read_parquet(ROOT / "outputs" / "preds_prophet_Bfixed.parquet"))
    sa = pd.read_parquet(ROOT / "outputs" / "preds_seasadj.parquet")
    parts.append(sa[sa["model"] == "DriftAdjSNaive"])
    df = pd.concat(parts, ignore_index=True)
    df = df[df["horizon"].isin(HORIZONS)]
    df = df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")
    df["category"] = df["series_id"].str.split("|").str[1]
    return df


def matched_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Пересечение (series_id, origin) по всем 4 моделям внутри горизонта — как 04."""
    keep = []
    for h, g in df.groupby("horizon"):
        sets = [set(zip(gg["series_id"], gg["origin"])) for _, gg in g.groupby("model")]
        inter = set.intersection(*sets)
        sel = g[[(s, o) in inter for s, o in zip(g["series_id"], g["origin"])]]
        keep.append(sel)
    return pd.concat(keep, ignore_index=True)


def loss_table(dm: pd.DataFrame, k_grid, label: str) -> pd.DataFrame:
    """Per-series средняя linlin-потеря -> pooled среднее по сериям (равный вес МО),
    + rel vs SeasonalNaive (суммы per-series, как RelMAE в 04)."""
    rows = []
    e = (dm["y_true"] - dm["y_pred"]).to_numpy(dtype=float)
    for k in k_grid:
        d = dm.assign(loss=linlin(e, k))
        ps = d.groupby(["model", "horizon", "series_id"], as_index=False)["loss"].mean()
        sn = ps[ps["model"] == "SeasonalNaive"].set_index(["horizon", "series_id"])["loss"]
        for (m, h), g in ps.groupby(["model", "horizon"]):
            num = g["loss"].sum()
            den = sn.loc[h].reindex(g["series_id"]).sum()
            rows.append({"slice": label, "model": m, "horizon": h, "k": k,
                         "linlin_rub": g["loss"].mean(),
                         "rel_vs_snaive": num / den,
                         "n_series": int(g.shape[0])})
    return pd.DataFrame(rows)


def kstar_sweep(pooled: pd.DataFrame) -> pd.DataFrame:
    """Чемпион (мин. потеря) на каждом k; k* = первое k со сменой чемпиона."""
    rows = []
    label = pooled["slice"].iloc[0]
    for h in HORIZONS:
        sub = pooled[pooled["horizon"] == h]
        champs = {}
        for k in K_SWEEP:
            sk = sub[np.isclose(sub["k"], k)]
            best = sk.loc[sk["linlin_rub"].idxmin()]
            champs[k] = (best["model"], best["linlin_rub"])
        base_champ = champs[float(K_SWEEP[0])][0]
        k_star = next((k for k in K_SWEEP if champs[float(k)][0] != base_champ), None)
        for k, (m, v) in champs.items():
            rows.append({"slice": label, "horizon": h, "k": float(k),
                         "pooled_champion": m, "champ_loss_rub": v,
                         "k_star": k_star})
    return pd.DataFrame(rows)


def quantile_shift(dm: pd.DataFrame, champ: str, k: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Смещение прогноза чемпиона на эмпирический tau=k/(1+k)-квантиль остатков,
    калиброванный expanding по origins < O (min 3), оценка на origin O."""
    tau = k / (1 + k)
    d = dm[dm["model"] == champ].copy()
    d["e"] = d["y_true"] - d["y_pred"]
    rows = []
    for (sid, h), g in d.groupby(["series_id", "horizon"]):
        g = g.sort_values("origin")
        origins = g["origin"].to_numpy()
        for i, o in enumerate(origins):
            if i < 3:  # минимум 3 прошлых origins для калибровки
                continue
            resid_cal = g["e"].to_numpy()[:i]
            shift = np.quantile(resid_cal, tau)
            e_true = g["e"].to_numpy()[i]
            rows.append({"series_id": sid, "horizon": h, "origin": o,
                         "e_point": e_true, "e_shift": e_true - shift,
                         "shift_rub": shift, "n_calib": i})
    ev = pd.DataFrame(rows)
    out = []
    for h, g in ev.groupby("horizon"):
        per_series = g.groupby("series_id").apply(
            lambda x: pd.Series({"point": linlin(x["e_point"].to_numpy(), k).mean(),
                                 "shifted": linlin(x["e_shift"].to_numpy(), k).mean()}),
            include_groups=False)
        w = paired_wilcoxon(per_series["point"], per_series["shifted"])
        out.append({"horizon": h, "k": k, "tau": tau,
                    "linlin_point_rub": per_series["point"].mean(),
                    "linlin_shifted_rub": per_series["shifted"].mean(),
                    "delta_pct": 100 * (per_series["shifted"].mean()
                                        / per_series["point"].mean() - 1),
                    "share_improved": float((per_series["shifted"]
                                             < per_series["point"]).mean()),
                    "median_shift_rub": float(g["shift_rub"].median()),
                    "n_series": int(per_series.shape[0]),
                    "n_origins": int(g["origin"].nunique()),
                    "wilcoxon_p": w.get("p", np.nan)})
    return pd.DataFrame(out), ev


def main():
    print("load...", flush=True)
    df = load_models()
    print(f"  {len(df)} строк; модели: {sorted(df.model.unique())}", flush=True)
    dm = matched_cells(df)
    n_cells = dm.groupby("horizon").size() // len(MODELS)
    print(f"  matched cells per horizon: {n_cells.to_dict()}", flush=True)

    # sanity: y уже в рублях (expm1 при записи в 02/03/08) — проверяем диапазон
    assert dm["y_true"].median() > 100, "y_true похож на лог-шкалу, стоп"

    print("loss tables...", flush=True)
    pooled = loss_table(dm, K_TABLE, "POOLED")
    allcat = loss_table(dm[dm["category"] == "Все категории"], K_TABLE, "Все категории")
    tab = pd.concat([pooled, allcat], ignore_index=True)
    tab.to_csv(ROOT / "outputs" / "rc1_o8_linlin.csv", index=False)
    for label in ["POOLED", "Все категории"]:
        sub = tab[tab["slice"] == label]
        piv = sub.pivot_table(index=["model", "horizon"], columns="k",
                              values="linlin_rub").round(0)
        print(f"\n=== {label}: linlin (руб/мес на ряд) ===\n{piv.to_string()}", flush=True)

    print("k* sweep (pooled + «Все категории»)...", flush=True)
    pooled_fine = loss_table(dm, K_SWEEP, "POOLED")
    allcat_fine = loss_table(dm[dm["category"] == "Все категории"], K_SWEEP,
                             "Все категории")
    ks = pd.concat([kstar_sweep(pooled_fine), kstar_sweep(allcat_fine)],
                   ignore_index=True)
    ks.to_csv(ROOT / "outputs" / "rc1_o8_kstar.csv", index=False)
    for sl in ["POOLED", "Все категории"]:
        for h in HORIZONS:
            sub = ks[(ks["slice"] == sl) & (ks["horizon"] == h)]
            k_star = sub["k_star"].iloc[0]
            ch = sub.drop_duplicates("pooled_champion")[["k", "pooled_champion"]]
            print(f"  [{sl}] h={h}: смены чемпиона:\n{ch.to_string(index=False)}\n"
                  f"    k* = {k_star}", flush=True)

    # Wilcoxon чемпион vs runner-up при k=2 (pooled)
    print("wilcoxon @k=2...", flush=True)
    for h in HORIZONS:
        sub = pooled_fine[(pooled_fine["horizon"] == h) & np.isclose(pooled_fine["k"], 2.0)]
        srt = sub.sort_values("linlin_rub")
        best, second = srt.iloc[0]["model"], srt.iloc[1]["model"]
        dh = dm[dm["horizon"] == h].copy()
        dh["loss"] = linlin((dh["y_true"] - dh["y_pred"]).to_numpy(dtype=float), 2.0)
        ps = dh.groupby(["model", "series_id"], as_index=False)["loss"].mean()
        a = ps[ps["model"] == best].set_index("series_id")["loss"]
        b = ps[ps["model"] == second].set_index("series_id")["loss"]
        w = paired_wilcoxon(a, b)
        print(f"  h={h}: {best} vs {second}: Wilcoxon p={w.get('p'):.3g}", flush=True)

    # диагностика: доля недопрогнозов (e>0) — почему лидер растёт с k
    dm2 = dm.copy()
    dm2["under"] = (dm2["y_true"] > dm2["y_pred"]).astype(float)
    uf = dm2.groupby(["model", "horizon"])["under"].mean().round(4)
    print("\nдоля недопрогнозов (y_true > y_pred):\n", uf.unstack().to_string(), flush=True)

    print("quantile shift @k=2 (DriftAdjSNaive + Prophet-B*)...", flush=True)
    outs = []
    for champ in ["DriftAdjSNaive", "Prophet-B*"]:
        for label, sub in [("POOLED", dm),
                           ("Все категории", dm[dm["category"] == "Все категории"])]:
            res, _ = quantile_shift(sub, champ, k=2.0)
            res.insert(0, "model", champ)
            res.insert(0, "slice", label)
            outs.append(res)
            print(f"\n[{champ} / {label}]\n", res.round(4).to_string(index=False), flush=True)
    shift_res_out = pd.concat(outs, ignore_index=True)
    shift_res_out.to_csv(ROOT / "outputs" / "rc1_o8_shift.csv", index=False)
    print("saved: outputs/rc1_o8_linlin.csv, rc1_o8_kstar.csv, rc1_o8_shift.csv", flush=True)


if __name__ == "__main__":
    main()
