"""Downstream ΔMAE v2: улучшает ли refit-от-брейка прогноз? (мост CPD→прогноз, R04 §2.3)

Фиксы аудита A11 P0-5 / A05 P0-3:
- брейки ищутся КАУЗАЛЬНО: для каждого origin o детектор (Window, порог из замороженной
  калибровки outputs/cpd_calibration_v2.csv) прогоняется только на y[:o]; раньше брейк
  со всей 24-месячной истории использовался для всех origins (look-ahead, расхождение
  датировки в 69% ячеек);
- fallback (post-break < MIN_POST → полный train) больше не маскируется под refit:
  ячейки помечены fallback=True и репортятся ОТДЕЛЬНО; «паритет при >4σ» rc0 был
  механическим артефактом (94% ячеек бакета — побитово одинаковые прогнозы);
- pooled ΔMAE (взвешен объёмом МО) дополнен median per-series ΔMAE, как требует R04;
- десезонирование — из src.cpd (ridge+Huber со шринкажем к профилю категории);
  замечание по протоколу: покатегорийный сезонный профиль оценён по всей панели
  (медиана тысяч рядов, вклад одного месяца одного ряда пренебрежим) — серийная часть
  оценки и детекция каузальны; это зафиксировано как допущение.

Два прогона детекции: рабочая точка бюджета (FAR≤2/1000) и, для сопоставимости с rc0,
legacy-точка Window@3.0 — с явными метками. Выходы с суффиксом _v2.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsforecast.models import AutoTheta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cpd import category_profiles, deseasonalize, det_window, mad_scale, K_SHRINK  # noqa: E402

HORIZONS = [1, 3, 6]
ORIGINS = list(range(12, 24))
MIN_POST = 8          # post-break ряд короче → fallback на полный train (R04), честно помечается
N_SERIES = 3000       # стратифицированная подвыборка (AutoTheta × origins × h — часы на полной)
SEED = 42


def stratified_sample(wide, cats, n, seed):
    """Страта = категория × квартиль медианного уровня ряда."""
    rng = np.random.default_rng(seed)
    med = np.nanmedian(wide, axis=1)
    q = pd.qcut(pd.Series(med), 4, labels=False)
    strata = pd.Series(cats) + "|" + pd.Series(q).astype(str)
    idx = []
    per = n // strata.nunique()
    for _, group in strata.groupby(strata):
        pool = np.where(strata == group[0])[0]
        idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False))
    return np.array(sorted(set(idx)))


def causal_last_break(y_raw, cat_prof, origin, pen):
    """Последний брейк строго внутри train: детектор видит только y[:origin]."""
    if origin < 10:  # Window(width=4) требует ≥ 2·width+1 точек для кандидатов
        return None, np.nan
    z = deseasonalize(y_raw[:origin], cat_prof, k=K_SHRINK)
    br = det_window(z, pen)
    if not br:
        return None, np.nan
    last = max(br)
    s = mad_scale(z)
    mag = abs(np.median(z[last:last + 3]) - np.median(z[max(0, last - 3):last])) / s
    return last, mag


def run_setting(wv, wr, sids, prof_arr, idx, pen, label):
    rows = []
    n_with_break = 0
    for c, i in enumerate(idx):
        y = wv[i]
        # каузальный детект на каждом origin отдельно
        br_by_o, mag_by_o = {}, {}
        for o in ORIGINS:
            b, m = causal_last_break(y, prof_arr[i], o, pen)
            if b is not None and o > b:
                br_by_o[o] = b
                mag_by_o[o] = m
        if br_by_o:
            n_with_break += 1
        for o, last_br in br_by_o.items():
            post_len = o - last_br
            is_fallback = post_len < MIN_POST
            start = 0 if is_fallback else last_br
            for h in HORIZONS:
                tm = o - 1 + h
                if tm > 23:
                    continue
                try:
                    p_full = AutoTheta().fit(y[:o]).predict(h=h)["mean"][-1]
                except Exception:
                    continue
                if is_fallback:
                    p_br = p_full  # fallback = полный train, прогноз совпадает по построению
                else:
                    try:
                        p_br = AutoTheta().fit(y[start:o]).predict(h=h)["mean"][-1]
                    except Exception:
                        p_br = p_full
                y_true = wr[i, tm]
                rows.append(dict(series_id=sids[i], origin=o, horizon=h, setting=label,
                                 ae_full=abs(np.expm1(p_full) - y_true),
                                 ae_refit=abs(np.expm1(p_br) - y_true),
                                 post_len=post_len, mag=mag_by_o[o], fallback=is_fallback))
        if (c + 1) % 500 == 0:
            print(f"  [{label}] {c+1}/{len(idx)}, {time.time()-T0:.0f}с", flush=True)
    return pd.DataFrame(rows), n_with_break


def summarize(df):
    """pooled + median per-series, раздельно по fallback/refit."""
    out = []
    for (st, h, fb), g in df.groupby(["setting", "horizon", "fallback"]):
        per_series = g.groupby("series_id").apply(
            lambda x: (x.ae_refit.sum() / max(x.ae_full.sum(), 1e-9) - 1), include_groups=False)
        out.append(dict(setting=st, horizon=h, fallback=bool(fb), n_cells=len(g),
                        n_series=g.series_id.nunique(),
                        dmae_pooled_pct=(g.ae_refit.mean() / max(g.ae_full.mean(), 1e-9) - 1) * 100,
                        dmae_median_series_pct=per_series.median() * 100,
                        share_refit_better=(g.ae_refit < g.ae_full).mean()))
    return pd.DataFrame(out)


def by_mag(df):
    """Бакеты величины брейка — ТОЛЬКО настоящие refit-ячейки (fallback исключён)."""
    d = df[~df.fallback].copy()
    if not len(d):
        return pd.DataFrame()
    d["mag_bucket"] = pd.cut(d["mag"], [0, 1, 2, 4, np.inf], labels=["<1σ", "1-2σ", "2-4σ", ">4σ"])
    return (d[d.horizon == 3].groupby(["setting", "mag_bucket"], observed=True)
            .apply(lambda g: pd.Series({"n": len(g),
                                        "dmae_pooled_pct": (g.ae_refit.mean() / max(g.ae_full.mean(), 1e-9) - 1) * 100,
                                        "share_refit_better": (g.ae_refit < g.ae_full).mean()}),
                   include_groups=False).reset_index())


def main():
    global T0
    T0 = time.time()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    panel["series_id"] = panel["territory_id"].astype(str) + "|" + panel["category"]
    wide = panel.pivot_table(index="series_id", columns="month_idx",
                             values="log_value").reindex(columns=range(24))
    wide_rub = panel.pivot_table(index="series_id", columns="month_idx",
                                 values="value").reindex(columns=range(24))
    cats = np.array([s.split("|")[1] for s in wide.index])
    profs = category_profiles(wide.to_numpy(), cats)
    prof_arr = np.array([profs[c] for c in cats])
    idx = stratified_sample(wide.to_numpy(), cats, N_SERIES, SEED)
    print(f"подвыборка: {len(idx)} рядов из {len(wide)}", flush=True)

    # замороженная рабочая точка из калибровки (06_run_cpd.py должен быть прогнан первым)
    cal = pd.read_csv(ROOT / "outputs" / "cpd_calibration_v2.csv")
    wrow = cal[(cal.method == "Window") & cal.is_working]
    pen_budget = float(wrow.param.iloc[0]) if len(wrow) else None
    if pen_budget is None:
        raise RuntimeError("нет замороженной точки Window в cpd_calibration_v2.csv — прогони 06")
    print(f"Window бюджетная точка: pen={pen_budget}", flush=True)

    wv, wr = wide.to_numpy(), wide_rub.to_numpy()
    sids = wide.index.to_numpy()
    frames = []
    for pen, label in [(pen_budget, "budget"), (3.0, "legacy_pen3")]:
        df, nb = run_setting(wv, wr, sids, prof_arr, idx, pen, label)
        print(f"[{label}] рядов с каузальным брейком: {nb}/{len(idx)}", flush=True)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(ROOT / "outputs" / "downstream_dmae_rows_v2.parquet")

    summ = summarize(df)
    summ.to_csv(ROOT / "outputs" / "downstream_dmae_v2.csv", index=False)
    bm = by_mag(df)
    bm.to_csv(ROOT / "outputs" / "downstream_dmae_by_mag_v2.csv", index=False)
    print(summ.round(3).to_string(index=False))
    print(bm.round(3).to_string(index=False))
    print(f"total {time.time()-T0:.0f}с")


if __name__ == "__main__":
    main()
