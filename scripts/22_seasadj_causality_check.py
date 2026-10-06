"""P0-проверка каузальности DriftAdjSNaive и SeasProfileHA (champion-кандидаты, 08_run_seasadj.py).

Три независимых теста:
1) HAND-RECOMPUTE: предсказания для 3 рядов × все 30 ячеек (origin × h) пересчитаны
   по спецификации заново (plain Python, без импорта кода 08) и сверены с
   outputs/preds_seasadj.parquet (допуск 1e-9 относительный).
2) FUTURE-MUTATION PROBE (из validation.yaml leakage_controls): все значения панели
   ПОСЛЕ origin заменены на +1e6 (log-space) → предсказания обязаны быть побитово
   теми же. Проверяется на origins 12, 18, 23 для ВСЕХ рядов.
3) DRIFT-AUDIT: pooled YoY-дрейф g(origin) пересчитан из train-only окна и сверен
   с тем, что фактически применён в parquet (выводится обратным ходом из разности
   DriftAdjSNaive и SeasonalNaive там, где tm-12 >= 0... вместо этого: прямое
   сравнение hand-g с g, восстановленным из log(pred)-log(y[tm-12])).

Выход: outputs/T5_causality_check.csv (по одной строке на тест) + печать вердикта.
Запуск: nice -n 10 .venv/bin/python scripts/22_seasadj_causality_check.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORIGINS = list(range(12, 24))
HORIZONS = [1, 3, 6, 12]
TOL = 1e-9


def build_wide(panel: pd.DataFrame):
    panel = panel.copy()
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    wide = panel.pivot_table(index="series_id", columns="month_idx", values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx", values="value").reindex(columns=range(24))
    reg = panel.groupby("series_id")["region_name"].first()
    return wide, wide_rub, reg


def hand_predict(wv: np.ndarray, reg_arr: np.ndarray, origin: int, h: int, model: str) -> np.ndarray:
    """Независимая реимплементация по спецификации (train = месяцы 0..origin-1)."""
    c = origin - 1
    assert 0 <= c <= 22
    tm = c + h
    if tm > 23:
        return None
    train = wv[:, : c + 1]  # строго <= origin
    med_all = np.nanmedian(train, axis=1)
    if model == "DriftAdjSNaive":
        g = np.nanmedian(wv[:, c] - wv[:, c - 12]) if c >= 12 else 0.0
        base = wv[:, tm - 12] if tm - 12 >= 0 else med_all
        p_log = base + g
    elif model == "SeasProfileHA":
        cal = np.arange(24) % 12
        mcols = [i for i in range(c + 1) if cal[i] == tm % 12]
        if mcols:
            mo_seas = np.nanmedian(wv[:, mcols], axis=1) - med_all
            # региональный профиль: медиана по рядам региона помесячно, затем по месяцам
            df_m = pd.DataFrame(wv[:, mcols]).groupby(reg_arr).median().median(axis=1)
            df_a = pd.DataFrame(train).groupby(reg_arr).median().median(axis=1)
            reg_ratio = (df_m - df_a).reindex(reg_arr).to_numpy()
            seas = (len(mcols) * np.nan_to_num(mo_seas, nan=0.0) + 2.0 * reg_ratio) / (len(mcols) + 2.0)
        else:
            seas = 0.0
        p_log = med_all + seas
    else:
        raise ValueError(model)
    return np.expm1(np.clip(p_log, 0, None))


def main():
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    wide, wide_rub, reg = build_wide(panel)
    wv = wide.to_numpy()
    reg_arr = reg.reindex(wide.index).to_numpy()
    preds = pd.read_parquet(ROOT / "outputs" / "preds_seasadj.parquet")
    results = []

    # --- Тест 1: hand-recompute 3 рядов × все ячейки ---
    cats = wide.index.str.split("|").str[1]
    picks = [
        wide.index[cats == "Маркетплейсы"][0],
        wide.index[cats == "Общественное питание"][0],
        wide.index[cats == "Все категории"][len(wide.index[cats == "Все категории"]) // 2],
    ]
    n_bad = 0
    n_cells = 0
    max_rel = 0.0
    for sid in picks:
        i = wide.index.get_loc(sid)
        for o in ORIGINS:
            for h in HORIZONS:
                for model in ["DriftAdjSNaive", "SeasProfileHA"]:
                    got = hand_predict(wv, reg_arr, o, h, model)
                    if got is None:
                        continue
                    ref = preds[(preds.series_id == sid) & (preds.origin == o) &
                                (preds.horizon == h) & (preds.model == model)]["y_pred"]
                    assert len(ref) == 1, (sid, o, h, model)
                    rel = abs(got[i] - ref.iloc[0]) / max(abs(ref.iloc[0]), 1e-12)
                    max_rel = max(max_rel, rel)
                    n_cells += 1
                    if rel > TOL:
                        n_bad += 1
                        print(f"  MISMATCH {sid} o={o} h={h} {model}: hand={got[i]:.6f} parquet={ref.iloc[0]:.6f}")
    results.append({"test": "hand_recompute_3series", "n_cells": n_cells,
                    "n_mismatch": n_bad, "max_rel_diff": max_rel,
                    "pass": n_bad == 0})

    # --- Тест 2: future-mutation probe (все ряды, origins 12/18/23) ---
    for o in [12, 18, 23]:
        c = o - 1
        wv_mut = wv.copy()
        wv_mut[:, c + 1:] = 1e6  # будущее «взорвано»
        worst = 0.0
        n_cmp = 0
        for h in HORIZONS:
            for model in ["DriftAdjSNaive", "SeasProfileHA"]:
                got = hand_predict(wv_mut, reg_arr, o, h, model)
                if got is None:
                    continue
                ref = preds[(preds.origin == o) & (preds.horizon == h) & (preds.model == model)]
                ref = ref.set_index("series_id").reindex(wide.index)["y_pred"].to_numpy()
                rel = np.nanmax(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-12))
                worst = max(worst, rel)
                n_cmp += int(np.isfinite(ref).sum())
        results.append({"test": f"future_mutation_origin{o}", "n_cells": n_cmp,
                        "n_mismatch": int(worst > TOL), "max_rel_diff": float(worst),
                        "pass": worst <= TOL})

    # --- Тест 3: drift-audit — g, восстановленный из parquet, равен train-only pooled YoY ---
    # Для tm-12>=0: log(pred) = log(y[tm-12]) + g  =>  g = log(pred) - log(y[tm-12]).
    drift_bad = 0
    drift_checked = 0
    for o in ORIGINS:
        c = o - 1
        g_true = np.nanmedian(wv[:, c] - wv[:, c - 12]) if c >= 12 else 0.0
        for h in HORIZONS:
            tm = c + h
            if tm > 23 or tm - 12 < 0:
                continue
            ref = preds[(preds.origin == o) & (preds.horizon == h) &
                        (preds.model == "DriftAdjSNaive")].set_index("series_id").reindex(wide.index)
            g_rec = np.log1p(ref["y_pred"].to_numpy()) - wv[:, tm - 12]
            # клип в 0 в log-space мог затереть g для рядов с p_log<0 — берем медиану по неклипнутым
            g_rec_med = np.nanmedian(g_rec)
            drift_checked += 1
            if abs(g_rec_med - g_true) > 1e-9:
                drift_bad += 1
                print(f"  DRIFT MISMATCH o={o} h={h}: recovered={g_rec_med:.8f} train-only={g_true:.8f}")
    results.append({"test": "drift_train_only_audit", "n_cells": drift_checked,
                    "n_mismatch": drift_bad, "max_rel_diff": np.nan,
                    "pass": drift_bad == 0})

    # --- Sanity: y_true в parquet == панель ---
    t = wide_rub.stack().rename("y").reset_index()
    t.columns = ["series_id", "test_idx", "y"]
    # test_idx = месяц прогноза = origin - 1 + h
    chk = preds.merge(t, left_on=["series_id"], right_on=["series_id"], how="left")
    chk = chk[chk["origin"] - 1 + chk["horizon"] == chk["test_idx"]]
    y_ok = np.allclose(chk["y_true"], chk["y"], rtol=0, atol=1e-9)
    results.append({"test": "y_true_matches_panel", "n_cells": len(chk),
                    "n_mismatch": int(not y_ok), "max_rel_diff": np.nan, "pass": bool(y_ok)})

    out = pd.DataFrame(results)
    out.to_csv(ROOT / "outputs" / "T5_causality_check.csv", index=False)
    print(out.to_string(index=False))
    print("VERDICT:", "CLEAN — утечки нет" if out["pass"].all() else "P0 LEAK — СМОТРИ СТРОКИ ВЫШЕ")


if __name__ == "__main__":
    main()
