"""Origin 12 (контекст 2023-01..2023-12 -> прогноз 2024-01..2024-12), два прогона:

A) log1p-таргет с полными квантилями -> калибровка 80% PI (coverage + pinball по
   горизонтам 1..12) + квантильные предсказания (q10/q50/q90, в рублях) для шок-радара.
B) F0-raw абляция: zero-shot на сырых рублях (без log) — контроль чувствительности
   к нормировке (уровни 13.9k-91k = зона исторического bf16-бага, пин >= 2.3.2).

Выходы: outputs/preds_chronos2_quant_o12.parquet,
        outputs/chronos_quantile_coverage.csv,
        outputs/preds_chronos2_zs_raw_o12.parquet,
        outputs/f0raw_vs_log_origin12.csv
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGIN = 12
HMAX = 12
Q_IDX = {0.1: 2, 0.5: 10, 0.9: 18}  # индексы в 21-уровневом выходе (pipe.quantiles)


def main():
    from chronos import Chronos2Pipeline

    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="mps")
    assert [pipe.quantiles[i] for i in Q_IDX.values()] == [0.1, 0.5, 0.9]

    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    panel["month_idx"] = (pd.PeriodIndex(panel["date"], freq="M").year - 2023) * 12 + \
                         pd.PeriodIndex(panel["date"], freq="M").month - 1
    wide_log = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    sids = wide_log.index.to_numpy()
    rub = wide_rub.to_numpy()
    y_true = rub[:, ORIGIN:ORIGIN + HMAX]  # (n, 12)

    out = {}
    for tag, mat in [("log", wide_log.to_numpy()), ("raw", rub)]:
        ctx = [mat[i, :ORIGIN] for i in range(len(mat))]
        t1 = time.time()
        fc = pipe.predict(ctx, prediction_length=HMAX)
        q = np.stack([np.asarray(f)[0][list(Q_IDX.values())] for f in fc])  # (n, 3, 12) порядок 0.1/0.5/0.9
        print(f"{tag}: {time.time()-t1:.0f}с", flush=True)
        out[tag] = q

    # --- A: квантили в рублях, калибровка ---
    ql = np.expm1(np.clip(out["log"], 0, None))  # clip в логах -> expm1, как 09
    recs = []
    for h in range(1, HMAX + 1):
        for j, sid in enumerate(sids):
            recs.append((sid, ORIGIN, h, y_true[j, h - 1], ql[j, 0, h - 1], ql[j, 1, h - 1], ql[j, 2, h - 1]))
    qp = pd.DataFrame(recs, columns=["series_id", "origin", "horizon", "y_true", "q10", "q50", "q90"])
    qp["model"] = "chronos2_zs_log"
    qp.to_parquet(ROOT / "outputs" / "preds_chronos2_quant_o12.parquet")

    cov_rows = []
    for h in range(1, HMAX + 1):
        g = qp[qp["horizon"] == h]
        cov = ((g["y_true"] >= g["q10"]) & (g["y_true"] <= g["q90"])).mean()
        def pinball(q, col):
            e = g["y_true"] - g[col]
            return np.maximum(q * e, (q - 1) * e).mean()
        cov_rows.append({"horizon": h, "n": len(g), "coverage_80PI": cov,
                         "pinball_q10": pinball(0.1, "q10"), "pinball_q50": pinball(0.5, "q50"),
                         "pinball_q90": pinball(0.9, "q90")})
    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(ROOT / "outputs" / "chronos_quantile_coverage.csv", index=False)
    print(cov_df.round(3))

    # --- B: raw vs log точечные ---
    med_raw = out["raw"][:, 1, :]  # q50 сырого прогона (уже в рублях)
    recs = []
    for h in range(1, HMAX + 1):
        for j, sid in enumerate(sids):
            recs.append((sid, ORIGIN, h, y_true[j, h - 1], med_raw[j, h - 1]))
    rp = pd.DataFrame(recs, columns=["series_id", "origin", "horizon", "y_true", "y_pred"])
    rp["model"] = "chronos2_zs_raw"
    rp[["series_id", "origin", "horizon", "y_true", "y_pred", "model"]].to_parquet(
        ROOT / "outputs" / "preds_chronos2_zs_raw_o12.parquet")

    comp = []
    for h in range(1, HMAX + 1):
        mae_log = np.abs(y_true[:, h - 1] - ql[:, 1, h - 1]).mean()
        mae_raw = np.abs(y_true[:, h - 1] - med_raw[:, h - 1]).mean()
        comp.append({"horizon": h, "MAE_log": mae_log, "MAE_raw": mae_raw, "raw_minus_log": mae_raw - mae_log})
    comp_df = pd.DataFrame(comp)
    comp_df.to_csv(ROOT / "outputs" / "f0raw_vs_log_origin12.csv", index=False)
    print(comp_df.round(1))


if __name__ == "__main__":
    main()
