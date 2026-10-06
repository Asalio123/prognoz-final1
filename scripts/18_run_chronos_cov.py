"""F1: Chronos-2 + ковариаты (К4). Та же сетка, что 09_run_chronos.py: origins 12..23,
h in {1,3,6,12}, таргет log1p, оценка в рублях после expm1(q50).

Past-only ковариаты (известны строго <= origin):
  - region_loo_growth: средний log-прирост м/м по региону внутри категории БЕЗ самого МО;
    значение в месяце t использует только value[t] и value[t-1] -> причинно чисто для любого origin.
  - cbr_keyrate: месячное среднее дневной ключевой ставки ЦБ (data/external/cbr_keyrate_2023_2024.csv).
Known-future ковариаты (есть и в future_df): календарь month sin/cos.

Leak-gate: для каждого origin assert'им, что LOO/ставка при t <= origin-1 идентичны
значениям, посчитанным по панели, усечённой на origin-1, и что контекст кончается на origin-1.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HORIZONS = [1, 3, 6, 12]
ORIGINS = list(range(12, 24))
N_MONTHS = 24


def month_idx_of(date_str: pd.Series) -> pd.Series:
    p = pd.PeriodIndex(date_str, freq="M")
    return (p.year - 2023) * 12 + p.month - 1


def load_keyrate_monthly() -> np.ndarray:
    kr = pd.read_csv(ROOT / "data" / "external" / "cbr_keyrate_2023_2024.csv", sep=";")
    kr["date"] = pd.to_datetime(kr["date"], format="%d.%m.%Y")
    kr["keyrate"] = kr["keyrate"].str.replace(",", ".").astype(float)
    kr["midx"] = (kr["date"].dt.year - 2023) * 12 + kr["date"].dt.month - 1
    monthly = kr.groupby("midx")["keyrate"].mean().reindex(range(N_MONTHS))
    assert monthly.notna().all(), "пропуски в месячной ставке ЦБ"
    return monthly.to_numpy()


def build_loo_growth(panel: pd.DataFrame) -> pd.DataFrame:
    """LOO-прирост по (region, category) на всей панели. Возвращает long-фрейм
    territory_id, category, month_idx, loo_growth. Значение при t зависит только от
    value[t] и value[t-1] (см. leak-gate в main)."""
    p = panel[["territory_id", "category", "region_name", "month_idx", "log_value"]].copy()
    p = p.sort_values(["territory_id", "category", "month_idx"])
    p["g"] = p.groupby(["territory_id", "category"])["log_value"].diff()
    grp = p.groupby(["region_name", "category", "month_idx"])["g"]
    stats = grp.agg(["sum", "count"]).rename(columns={"sum": "g_sum", "count": "g_n"})
    p = p.merge(stats, left_on=["region_name", "category", "month_idx"], right_index=True)
    glob = p.groupby(["category", "month_idx"])["g"].mean().rename("g_glob")
    p = p.merge(glob, left_on=["category", "month_idx"], right_index=True)
    loo = (p["g_sum"] - p["g"]) / (p["g_n"] - 1)
    p["loo_growth"] = np.where(p["g_n"] > 1, loo, p["g_glob"])
    p["loo_growth"] = p["loo_growth"].fillna(0.0)  # month_idx 0: прироста нет
    return p[["territory_id", "category", "month_idx", "loo_growth"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default="amazon/chronos-2")
    ap.add_argument("--tag", default="chronos2_cov_log")
    args = ap.parse_args()

    from chronos import Chronos2Pipeline

    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained(args.model, device_map=args.device)
    print(f"load {time.time()-t0:.0f}с, device={args.device}", flush=True)

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = month_idx_of(panel["date"])
    panel["ts"] = pd.PeriodIndex(panel["date"], freq="M").to_timestamp()
    if args.limit:
        keep = panel["series_id"].unique()[: args.limit]
        panel = panel[panel["series_id"].isin(keep)]

    loo = build_loo_growth(panel)
    keyrate = load_keyrate_monthly()

    df = panel.merge(loo, on=["territory_id", "category", "month_idx"], how="left")
    df["cbr_keyrate"] = keyrate[df["month_idx"].to_numpy()]
    m = df["ts"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * m / 12)
    df["month_cos"] = np.cos(2 * np.pi * m / 12)
    assert df[["loo_growth", "cbr_keyrate"]].notna().all().all()

    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(N_MONTHS))

    # --- leak-gate: LOO, посчитанный по полной панели, обязан совпадать с LOO по усечённой ---
    for o in ORIGINS:
        loo_trunc = build_loo_growth(panel[panel["month_idx"] <= o - 1])
        chk = loo[loo["month_idx"] <= o - 1].merge(
            loo_trunc, on=["territory_id", "category", "month_idx"], suffixes=("", "_t")
        )
        assert np.allclose(chk["loo_growth"], chk["loo_growth_t"], atol=1e-12), f"LOO leak at origin {o}"
    print("leak-gate LOO: OK (12 origins)", flush=True)

    base_cols = ["series_id", "ts", "month_idx", "log_value", "loo_growth", "cbr_keyrate", "month_sin", "month_cos"]
    preds = []
    for o in ORIGINS:
        hmax = N_MONTHS - o
        ctx = df[df["month_idx"] <= o - 1][base_cols].rename(
            columns={"series_id": "item_id", "ts": "timestamp", "log_value": "target"}
        )
        # leak-gate: контекст строго < origin; будущее — только календарь
        assert ctx["month_idx"].max() == o - 1
        ctx = ctx.sort_values(["item_id", "timestamp"]).drop(columns=["month_idx"])
        fut_idx = np.arange(o, N_MONTHS)
        fut = pd.DataFrame({
            "item_id": np.repeat(ctx["item_id"].unique(), hmax),
            "timestamp": np.tile(pd.date_range("2023-01-01", periods=N_MONTHS, freq="MS")[fut_idx], ctx["item_id"].nunique()),
        })
        fm = fut["timestamp"].dt.month
        fut["month_sin"] = np.sin(2 * np.pi * fm / 12)
        fut["month_cos"] = np.cos(2 * np.pi * fm / 12)
        assert set(fut.columns) - {"item_id", "timestamp"} == {"month_sin", "month_cos"}, "known-future только календарь"

        t1 = time.time()
        fc = pipe.predict_df(
            ctx, future_df=fut, target="target",
            prediction_length=hmax, quantile_levels=[0.5], batch_size=256, freq="MS",
        )
        med = fc.pivot_table(index="item_id", columns="timestamp", values="0.5")
        med = med.reindex(columns=pd.date_range("2023-01-01", periods=N_MONTHS, freq="MS")[fut_idx])
        med_arr = med.to_numpy()  # (n_series, hmax), порядок item_id как в ctx.unique()
        sids = med.index.to_numpy()
        for h in HORIZONS:
            if h <= hmax:
                preds.append(pd.DataFrame({"series_id": sids, "origin": o, "horizon": h,
                                           "y_pred": np.expm1(np.clip(med_arr[:, h - 1], 0, None)),
                                           "model": args.tag, "test_idx": o + h - 1}))
        print(f"origin {o}: {time.time()-t1:.0f}с ({(time.time()-t1)/len(sids)*1000:.1f} мс/ряд)", flush=True)

    pred = pd.concat(preds, ignore_index=True)
    truth = wide_rub.stack().rename("y_true").reset_index().rename(columns={"month_idx": "test_idx"})
    pred = pred.merge(truth, on=["series_id", "test_idx"], how="left")
    assert pred["y_true"].notna().all() and pred["y_pred"].notna().all()
    out = ROOT / "outputs" / f"preds_{args.tag}{'_lim' + str(args.limit) if args.limit else ''}.parquet"
    pred[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(out)
    print("saved", out, len(pred), "rows")


if __name__ == "__main__":
    main()
