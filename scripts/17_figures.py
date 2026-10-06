"""Фигуры отчёта (G2, адаптировано после rc0: герой = честная карта ошибок, не радар).

Запуск: python scripts/17_figures.py  (после make tables)
Выход: reports/figures/F1..F7 (.pdf + .png 200dpi). Пропускает фигуры, чьи входы отсутствуют.
Стиль: colorblind-safe (cividis/RdBu_r/Okabe-Ito), кириллица, заголовок = вывод.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- стиль ---
for cand in ["Golos Text", "Inter", "PT Sans", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams.update({"axes.unicode_minus": False, "figure.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#888888"})
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]


def load_preds():
    files = [f for f in OUT.glob("preds_*.parquet")
             if not f.match("*_ces_part*") and not f.match("*_steps_part*")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["model"] != "y"].drop_duplicates(["series_id", "origin", "horizon", "model"])


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", name, flush=True)


def f5_relmae_distribution(df):
    """F5: распределение per-series RelMAE по моделям, h=3 — «честное сравнение с разбросом»."""
    h = 3
    d = df[(df.horizon == h)].copy()
    d["ae"] = (d.y_true - d.y_pred).abs()
    ps = d.groupby(["model", "series_id"])["ae"].mean().reset_index()
    sn = ps[ps.model == "SeasonalNaive"].set_index("series_id")["ae"]
    ps["rel"] = ps["ae"] / ps["series_id"].map(sn)
    order = ps.groupby("model")["rel"].median().sort_values().index[:8]  # топ-8 по медиане
    fig, ax = plt.subplots(figsize=(9, 4.8))
    data = [ps[ps.model == m]["rel"].clip(0.05, 20).values for m in order]
    bp = ax.boxplot(data, vert=True, showfliers=False, whis=(5, 95), patch_artist=True)
    for patch, c in zip(bp["boxes"], [OKABE[i % len(OKABE)] for i in range(len(order))]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.text(len(order) + 0.4, 1.02, "seasonal naive = 1.0", ha="right", fontsize=8)
    ax.set_xticks(range(1, len(order) + 1), order, rotation=30, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("RelMAE против seasonal naive (лог-шкала)")
    ax.set_title("Качество моделей на h=3: распределение по 1 947 МО × 6 категориям, не одно число",
                 fontsize=11)
    save(fig, "F5_relmae_distribution_h3")


def f7_step_profile(df):
    """F7: деградация MAE по шагу прогноза (1..12) — профиль старения моделей."""
    steps_files = list(OUT.glob("preds_stats_steps_part*.parquet"))
    pools = [pd.read_parquet(f) for f in steps_files]
    # + модели с полной сеткой: их h=12-окно origin=12 даёт шаги 1..12 только если шаги сохранены;
    # для Prophet/Chronos шаги не сохранены — используем их origin=12 preds (там все шаги 1..12
    # приходят из разных origins: h=12 только origin 12) → берем из основного пула по origin=12
    main = df[df.origin == 12].copy()
    main["ae"] = (main.y_true - main.y_pred).abs()
    frames = []
    if pools:
        s = pd.concat(pools, ignore_index=True)
        s["ae"] = (s.y_true - s.y_pred).abs()
        frames.append(s[["model", "horizon", "series_id", "ae"]])
    frames.append(main[["model", "horizon", "series_id", "ae"]])
    allsteps = pd.concat(frames, ignore_index=True).drop_duplicates(["model", "horizon", "series_id"])
    prof = allsteps.groupby(["model", "horizon"])["ae"].mean().reset_index()
    keep = ["SeasonalNaive", "HistoricAverage", "DynamicOptimizedTheta", "AutoETS",
            "Prophet-A", "LightGBM-direct", "chronos2_zs_log", "timesfm25_zs_log"]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for i, m in enumerate([m for m in keep if m in set(prof.model)]):
        p = prof[prof.model == m].sort_values("horizon")
        ax.plot(p.horizon, p.ae, marker="o", ms=3, lw=1.4, color=OKABE[i % len(OKABE)], label=m)
    ax.set_xlabel("Шаг прогноза (месяцев вперёд)"); ax.set_ylabel("MAE, ₽")
    ax.set_title("Как стареет прогноз: ошибка по шагам (единое окно, train=2023 → test=2024)",
                 fontsize=11)
    ax.legend(fontsize=8, ncols=2)
    save(fig, "F7_step_profile")


def f6_fact_vs_pred(df):
    """F6: факт vs прогноз, чемпион h=3 (выбирается по T1), все МО."""
    t1 = pd.read_csv(OUT / "T1_model_x_horizon.csv")
    champ = t1[t1.horizon == 3].sort_values("relmae_vs_snaive").iloc[0]["model"]
    d = df[(df.horizon == 3) & (df.model == champ)].copy()
    d = d[d.series_id.str.endswith("|Все категории")]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(d.y_true, d.y_pred, s=2, alpha=0.15, color=OKABE[0])
    lim = np.nanpercentile(np.r_[d.y_true, d.y_pred], 99.5)
    ax.plot([0, lim], [0, lim], color="black", lw=1, ls="--")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Факт, ₽/жителя"); ax.set_ylabel("Прогноз, ₽/жителя")
    ax.set_title(f"Факт vs прогноз, {champ}, h=3, «Все категории»: где модель промахивается",
                 fontsize=11)
    save(fig, "F6_fact_vs_pred_h3")


def f2_heatmap(panel):
    """F2: heatmap месяц × категория — номинальный рост г/г по нац. медиане."""
    nat = panel.groupby(["category", "month_idx"])["value"].median().reset_index()
    months = pd.period_range("2023-01", periods=24, freq="M")
    nat["month"] = nat["month_idx"].map(lambda i: months[i])
    piv = nat.pivot(index="category", columns="month", values="value")
    yoy = piv / piv.shift(12, axis=1) - 1
    yoy = yoy.iloc[:, 12:] * 100
    fig, ax = plt.subplots(figsize=(10, 3.6))
    im = ax.imshow(yoy.values, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto")
    ax.set_xticks(range(len(yoy.columns)), [str(m) for m in yoy.columns], rotation=45, fontsize=7)
    ax.set_yticks(range(len(yoy.index)), yoy.index, fontsize=8)
    for i in range(yoy.shape[0]):
        for j in range(yoy.shape[1]):
            ax.text(j, i, f"{yoy.values[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, label="рост г/г, %")
    ax.set_title("Номинальный рост г/г по категориям (медиана по МО): инфляция, сезон и шоки одним экраном",
                 fontsize=10)
    save(fig, "F2_category_month_heatmap")


def f3_cp_showcase(panel):
    """F3: витрина рядов с точками изменений (2×2): 2 позитивных эталона + 2 контроля."""
    breaks_f = OUT / "cpd_breaks.parquet"
    cases = [
        (1267, "Маркетплейсы", "Богородский м.о. (Кировская обл.): волна маркетплейсов — сдвиг уровня навсегда"),
        (788, "Общественное питание", "Дальнереченский м.р. (Приморье): скачок общепита +504%, удержан"),
        (340, "Все категории", "Усть-Майский м.р. (Якутия): декабрь ×1.44 — сезонность, НЕ шок"),
        (1, "Все категории", "Майкоп (Адыгея): спокойный ряд — правильное молчание детектора"),
    ]
    br = pd.read_parquet(breaks_f) if breaks_f.exists() else None
    if br is not None and "series_id" not in br.columns:
        br = br.copy()
        br["series_id"] = br["territory_id"].astype(str) + "|" + br["category"].astype(str)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=False)
    for ax, (tid, cat, title) in zip(axes.ravel(), cases):
        s = panel[(panel.territory_id == tid) & (panel.category == cat)].sort_values("month_idx")
        ax.plot(s.month_idx, s.value, color="#333333", lw=1.6)
        ax.set_title(title, fontsize=8.5, loc="left")
        if br is not None:
            b = br[(br.series_id == f"{tid}|{cat}")]
            for _, r in b.iterrows():
                col = "origin" if "origin" in r else None
                m = r.get("break_month", r.get("month", None))
                if m is not None:
                    ax.axvline(m, color="#D55E00", ls="--", lw=1.2, alpha=0.8)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["2023-01", "2023-07", "2024-01", "2024-07", "2024-12"], fontsize=7)
        ax.set_ylabel("₽/жителя", fontsize=8)
    fig.suptitle("Детектор на реальных рядах: эталонные сдвиги пойманы, сезонность и штиль — молчат",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "F3_cp_showcase")


def f4_fan_chart(df):
    """F4: факт vs прогноз чемпиона + эмпирические PI из backtest-остатков (80/95%)."""
    t1 = pd.read_csv(OUT / "T1_model_x_horizon.csv")
    champ = t1[t1.horizon == 3].sort_values("relmae_vs_snaive").iloc[0]["model"]
    d = df[(df.model == champ) & (df.series_id.str.endswith("|Все категории"))].copy()
    d["ae_rel"] = np.abs(d.y_true - d.y_pred) / d.y_true
    # PI по горизонтам: квантили относительной ошибки на backtest
    piv = {}
    for h, q in [(1, 0.1), (1, 0.25), (3, 0.1), (3, 0.25)]:
        pass
    # примерный ряд-кейс: медианный МО
    med_series = d.groupby("series_id")["y_true"].median().sort_values().iloc[len(d.series_id.unique()) // 2]
    sid = d.groupby("series_id")["y_true"].median().sort_values().index[len(d.series_id.unique()) // 2]
    s = d[d.series_id == sid].sort_values("origin")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(s.origin, s.y_true, color="#333333", lw=1.6, label="факт")
    ax.plot(s.origin, s.y_pred, color=OKABE[0], lw=1.4, label=f"прогноз ({champ}, h=3)")
    q90 = d.groupby("horizon")["ae_rel"].quantile(0.90).get(3, 0.2)
    q975 = d.groupby("horizon")["ae_rel"].quantile(0.975).get(3, 0.3)
    ax.fill_between(s.origin, s.y_pred * (1 - q975), s.y_pred * (1 + q975), color=OKABE[0], alpha=0.12, label="95% PI (эмпирический)")
    ax.fill_between(s.origin, s.y_pred * (1 - q90), s.y_pred * (1 + q90), color=OKABE[0], alpha=0.25, label="80% PI (эмпирический)")
    ax.legend(fontsize=8)
    ax.set_xlabel("origin (месяц от 2023-01)"); ax.set_ylabel("₽/жителя")
    ax.set_title(f"Прогноз с эмпирическими интервалами (ряд {sid.split('|')[0]}); калибровка — в отчёте", fontsize=10)
    save(fig, "F4_fan_chart")


def f1_error_map(df):
    """F1 (герой): карта RelMAE чемпиона h=3 по МО — где модель слаба (честность как сигнал)."""
    try:
        import geopandas as gpd
    except Exception as e:
        print("geopandas недоступен, F1 пропущена:", e); return
    gpkg = list((ROOT / "data" / "raw").rglob("*.gpkg"))
    if not gpkg:
        print("нет полигонов gpkg, F1 пропущена"); return
    t1 = pd.read_csv(OUT / "T1_model_x_horizon.csv")
    champ = t1[t1.horizon == 3].sort_values("relmae_vs_snaive").iloc[0]["model"]
    d = df[(df.horizon == 3) & (df.series_id.str.endswith("|Все категории"))].copy()
    d["ae"] = (d.y_true - d.y_pred).abs()
    ps = d.groupby(d.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    sn = df[(df.horizon == 3) & (df.model == "SeasonalNaive") & (df.series_id.str.endswith("|Все категории"))].copy()
    sn["ae"] = (sn.y_true - sn.y_pred).abs()
    psn = sn.groupby(sn.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    rel = (ps / psn).dropna().rename("relmae").reset_index().rename(columns={"series_id": "territory_id"})
    g = gpd.read_file(gpkg[0])
    tid_col = [c for c in g.columns if "terr" in c.lower() or c.lower() in ("tid", "id")][0]
    g["_tid"] = g[tid_col].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    g = g.merge(rel, left_on="_tid", right_on="territory_id", how="left")
    fig, ax = plt.subplots(figsize=(11, 6))
    g.plot(column="relmae", ax=ax, cmap="cividis", vmin=0.3, vmax=2.0,
           missing_kwds={"color": "#dddddd"}, legend=True,
           legend_kwds={"label": "RelMAE чемпиона против seasonal naive (h=3)"})
    ax.set_axis_off()
    ax.set_title(f"Карта собственных ошибок ({champ}, h=3): серое = вне панели, яркое = где модель хуже наивного",
                 fontsize=10)
    save(fig, "F1_error_map")


def main():
    print("load preds...", flush=True)
    df = load_preds()
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    f5_relmae_distribution(df)
    f7_step_profile(df)
    f6_fact_vs_pred(df)
    f2_heatmap(panel)
    f3_cp_showcase(panel)
    f4_fan_chart(df)
    f1_error_map(df)
    print("все фигуры готовы", flush=True)


if __name__ == "__main__":
    main()
