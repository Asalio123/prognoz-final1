"""Фигуры отчёта (G2, адаптировано после rc0: герой = честная карта ошибок, не радар).

Запуск: python scripts/17_figures.py  (после make tables)
Выход: reports/figures/F1..F7 (.pdf + .png 200dpi). Пропускает фигуры, чьи входы отсутствуют.

Стиль (светлое ядро, фигуры всегда на светлой подложке — даже для тёмной деки):
фон #FFFDF7 (бумага), текст #1D2530, сетка только горизонтальная #E9E4D3,
спайны верх/право убраны. Акценты: наши #233C67, чужие #9AA1AB,
негатив #A33327, позитив #2F6B4F. Заголовок = вывод, подзаголовок = метод,
футер = источник. Шрифт PT Sans (кириллица), fallback DejaVu Sans.
Данные и агрегации не менялись — правится только отрисовка.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# --- токены светлого ядра ---
BG = "#FFFDF7"      # бумага
INK = "#1D2530"     # основной текст
MUTED = "#5B6472"   # подзаголовок
FAINT = "#8B8574"   # футер
GRID = "#E9E4D3"    # сетка (только горизонтальная)
OURS = "#233C67"    # наши модели
THEIR = "#9AA1AB"   # baseline / чужие
NEG = "#A33327"     # хуже / негатив
POS = "#2F6B4F"     # лучше / позитив

FULL = (11.0, 5.3)   # full-width
HALF = (5.9, 4.2)    # half-width

# diverging-шкала для величин с естественным центром: синий (лучше/ниже) →
# бумага (центр) → бордовый (хуже/выше). В грейскейле: оба края тёмные,
# центр светлый, missing отличён штриховкой.
DIVERGE = LinearSegmentedColormap.from_list("paper_div", [OURS, "#F3EEE0", NEG])

ALBERS = ("+proj=aea +lat_1=50 +lat_2=70 +lat_0=60 +lon_0=100 "
          "+datum=WGS84 +units=m +no_defs")

# --- шрифт: PT Sans (кириллица), fallback DejaVu Sans ---
for cand in ["PT Sans", "DejaVu Sans"]:
    try:
        matplotlib.font_manager.findfont(cand, fallback_to_default=False)
        plt.rcParams["font.family"] = cand
        break
    except Exception:
        continue
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.dpi": 200,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#B9B2A0", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
    "font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12.5,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
})

# Okabe-Ito без жёлтого #F0E442 (невидим на светлом и пропадает в грейскейле)
OKABE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def hgrid(ax):
    """Сетка только горизонтальная, под данными."""
    ax.grid(axis="y", color=GRID, lw=0.9)
    ax.set_axisbelow(True)


def head(fig, title, subtitle=None):
    """Заголовок-вывод + подзаголовок-метод, левый край.
    Подзаголовок опускается на число строк заголовка (высоты считаются от figsize)."""
    h_in = fig.get_size_inches()[1]
    lh = 15.5 * 1.25 / (h_in * 72)  # доля высоты фигуры на строку заголовка
    n = title.count("\n") + 1
    fig.text(0.012, 0.978, title, ha="left", va="top",
             fontsize=15.5, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.012, 0.978 - n * lh - 0.45 * lh, subtitle, ha="left", va="top",
                 fontsize=10.5, color=MUTED)


def foot(fig, text):
    fig.text(0.012, 0.012, text, ha="left", va="bottom", fontsize=8.5, color=FAINT)


SOURCE = "Источник: СберИндекс — помесячные ряды 2 016 муниципальных образований, 2023–2024."


def load_preds():
    files = [f for f in OUT.glob("preds_*.parquet")
             if not f.match("*_ces_part*") and not f.match("*_steps_part*")
             and not f.match("*_quant_o12*") and not f.match("*_raw_o12*")
             and not f.match("*coldstart*")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df[df["model"] != "y"].drop_duplicates(["series_id", "origin", "horizon", "model"])


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", facecolor=BG)
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight", dpi=200, facecolor=BG)
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
    meds = ps.groupby("model")["rel"].median()

    # наши конфигурации (собраны/обучены в работе) против готовых baseline
    ours = {"DriftAdjSNaive", "SeasProfileHA", "Ensemble-median3",
            "LightGBM-direct", "LightGBM-direct-news", "LightGBM-direct-news_v2",
            "LightGBM-direct_s271", "chronos2_cov_log"}
    best, best_med = order[0], meds[order[0]]

    fig = plt.figure(figsize=FULL)
    ax = fig.add_axes([0.175, 0.15, 0.72, 0.665])
    # лучшая медиана — верхняя строка
    models = list(order)[::-1]
    data = [ps[ps.model == m]["rel"].clip(0.05, 20).values for m in models]
    bp = ax.boxplot(data, orientation="horizontal", showfliers=False, whis=(5, 95), patch_artist=True,
                    widths=0.62, medianprops={"color": INK, "lw": 1.6},
                    whiskerprops={"lw": 1.1}, capprops={"lw": 0})
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(OURS if m in ours else THEIR)
        patch.set_alpha(0.9 if m in ours else 0.65)
        patch.set_edgecolor("none")
    for i, m in enumerate(models):
        ax.text(24, i + 1, f"{meds[m]:.2f}", va="center", ha="left", fontsize=10.5,
                color=OURS if m in ours else MUTED)
    ax.axvline(1.0, color=NEG, ls="--", lw=1.4, zorder=0)
    ax.text(1.03, len(models) + 0.65, "1.0 = сезонный наивный", fontsize=10.5,
            color=NEG, va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
    ax.set_xticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
    ax.set_xlim(0.09, 60)
    ax.set_yticks(range(1, len(models) + 1), models, fontsize=11.5)
    ax.set_ylim(0.4, len(models) + 1.15)
    ax.set_xlabel("RelMAE = MAE модели / MAE сезонного наивного на том же ряду (лог-шкала)")
    ax.grid(axis="x", color=GRID, lw=0.9)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(facecolor=OURS, label="наши конфигурации"),
                        Patch(facecolor=THEIR, alpha=0.65, label="baseline и статистика")],
               loc="upper right", bbox_to_anchor=(0.995, 0.865), fontsize=10.5, frameon=False)
    head(fig,
         f"На полном пуле сильнейший — {best} (медиана RelMAE {best_med:.2f});\n"
         "у всех моделей хвост ошибок тянется за ×2",
         "Распределение RelMAE по 11 682 рядам (1 947 МО × 6 категорий), h=3; "
         "бокс = квартили, ус = 5–95 перцентили, выбросы скрыты")
    foot(fig, SOURCE)
    save(fig, "F5_relmae_distribution_h3")


def f7_step_profile(df):
    """F7: деградация MAE по шагу прогноза (1..12) — профиль старения моделей.
    Светлая версия rc1_s4-ремейка: 6 линий, Okabe-Ito, прямые подписи у концов."""
    steps_files = list(OUT.glob("preds_stats_steps_part*.parquet"))
    pools = [pd.read_parquet(f) for f in steps_files]
    # для Prophet/Chronos шаги не сохранены — берём их origin=12 preds из основного пула
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

    colors = {"Prophet-A": "#CC79A7", "AutoETS": "#D55E00", "HistoricAverage": "#E69F00",
              "LightGBM-direct": "#56B4E9", "chronos2_zs_log": "#009E73",
              "SeasonalNaive": "#000000"}
    keep = [m for m in ["SeasonalNaive", "HistoricAverage", "DynamicOptimizedTheta", "AutoETS",
                        "Prophet-A", "LightGBM-direct", "chronos2_zs_log", "timesfm25_zs_log"]
            if m in colors and m in set(prof.model)]

    fig = plt.figure(figsize=FULL)
    ax = fig.add_axes([0.075, 0.13, 0.80, 0.68])
    for m in keep:
        p = prof[prof.model == m].sort_values("horizon")
        if m == "SeasonalNaive":
            ax.plot(p.horizon, p.ae, marker="s", ms=5, lw=2.2, color=colors[m],
                    ls="--", zorder=5)
        else:
            ax.plot(p.horizon, p.ae, marker="o", ms=4.5, lw=2.0, color=colors[m])
        label = m.replace("_zs_log", "")
        if m == "LightGBM-direct":
            # линия обрывается на h=6 — подпись у пика h=1, где вокруг пусто
            ax.annotate(label, (p.horizon.iloc[0], p.ae.iloc[0]),
                        xytext=(8, 8), textcoords="offset points", fontsize=11.5,
                        color=colors[m], va="bottom", fontweight="bold")
        else:
            ax.annotate(label, (p.horizon.iloc[-1], p.ae.iloc[-1]),
                        xytext=(7, 0), textcoords="offset points", fontsize=11.5,
                        color=colors[m], va="center", fontweight="bold")
    ax.set_xlim(0.6, 14.2)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Шаг прогноза (месяцев вперёд)")
    ax.set_ylabel("MAE, ₽/жителя")
    hgrid(ax)
    head(fig,
         "Ошибка foundation-моделей к 12-му месяцу удваивается,\n"
         "статистика и Prophet-A стареют мягко",
         "Средний MAE по рядам панели на каждом шаге прогноза; единое окно train=2023, test=2024")
    foot(fig, SOURCE)
    save(fig, "F7_step_profile")


def f6_fact_vs_pred(df):
    """F6: факт vs прогноз, чемпион h=3 (выбирается по T1), все МО."""
    t1 = pd.read_csv(OUT / "T1_model_x_horizon.csv")
    champ = t1[t1.horizon == 3].sort_values("relmae_vs_snaive").iloc[0]["model"]
    d = df[(df.horizon == 3) & (df.model == champ)].copy()
    d = d[d.series_id.str.endswith("|Все категории")]
    r = d.y_pred / d.y_true
    within25 = ((r >= 0.8) & (r <= 1.25)).mean()

    fig = plt.figure(figsize=HALF)
    ax = fig.add_axes([0.155, 0.175, 0.80, 0.585])
    ax.scatter(d.y_true, d.y_pred, s=3, alpha=0.12, color=OURS, edgecolors="none")
    lim = np.nanpercentile(np.r_[d.y_true, d.y_pred], 99.5)
    ax.plot([0, lim], [0, lim], color=NEG, lw=1.3, ls="--", zorder=5)
    ax.annotate("идеал: прогноз = факт", (lim * 0.28, lim * 0.28),
                xytext=(0, 7), textcoords="offset points", fontsize=10,
                color=NEG, rotation=38, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Факт, ₽/жителя (лог-шкала)")
    ax.set_ylabel("Прогноз, ₽/жителя (лог-шкала)")
    hgrid(ax)
    head(fig,
         f"Прогноз попадает в ±25% от факта\nв {within25 * 100:.0f}% точек",
         f"{champ}, h=3, «Все категории»; точка = МО × месяц прогноза")
    foot(fig, SOURCE)
    save(fig, "F6_fact_vs_pred_h3")


def f2_heatmap(panel):
    """F2: heatmap месяц × категория — номинальный рост г/г по нац. медиане."""
    nat = panel.groupby(["category", "month_idx"])["value"].median().reset_index()
    months = pd.period_range("2023-01", periods=24, freq="M")
    nat["month"] = nat["month_idx"].map(lambda i: months[i])
    piv = nat.pivot(index="category", columns="month", values="value")
    yoy = piv / piv.shift(12, axis=1) - 1
    yoy = yoy.iloc[:, 12:] * 100
    center = np.nanmedian(yoy.values)  # типичный номинальный рост года — центр дивергенции
    span = max(abs(np.nanpercentile(yoy.values, 5) - center), abs(np.nanpercentile(yoy.values, 95) - center))
    mp = yoy.loc["Маркетплейсы"]

    fig = plt.figure(figsize=FULL)
    ax = fig.add_axes([0.155, 0.20, 0.70, 0.565])
    norm = TwoSlopeNorm(vmin=center - span, vcenter=center, vmax=center + span)
    im = ax.imshow(yoy.values, cmap=DIVERGE, norm=norm, aspect="auto")
    ax.set_xticks(range(len(yoy.columns)))
    ax.set_xticklabels(["Янв", "Фев", "Мар", "Апр", "Май", "Июн",
                        "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"], fontsize=11.5)
    ax.set_xlabel("месяц 2024 года")
    ax.set_yticks(range(len(yoy.index)), yoy.index, fontsize=12)
    for i in range(yoy.shape[0]):
        for j in range(yoy.shape[1]):
            v = yoy.values[i, j]
            ax.text(j, i, f"{v:+.0f}", ha="center", va="center", fontsize=10.5,
                    color="white" if abs(v - center) > 0.62 * span else INK)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.155, 0.085, 0.42, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    ticks = [center - span, center, center + span]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{ticks[0]:+.0f}%", f"{center:+.0f}% — типичный рост года",
                       f"{ticks[2]:+.0f}%"], fontsize=9.5)
    cb.outline.set_visible(False)
    head(fig,
         f"Маркетплейсы растут на {mp.min():+.0f}…{mp.max():+.0f}% каждый месяц —\n"
         "это тренд, а не сезонная волна",
         "Номинальный рост г/г по категориям, медиана по МО; "
         "красное = выше типичного роста года, синее = ниже")
    foot(fig, SOURCE)
    save(fig, "F2_category_month_heatmap")


def f3_cp_showcase(panel):
    """F3: витрина рядов с истинными событиями (2×2): 2 позитивных эталона + 2 контроля.
    Линия = истинный месяц события; в подзаголовке — результат детектора (лаги честно)."""
    cases = [
        (1267, "Маркетплейсы", 13, "Богородский м.о. (Кировская обл.): +402% навсегда; PELT/BinSeg ловят с лагом 2 мес"),
        (788, "Общественное питание", 5, "Дальнереченский м.р. (Приморье): +504%, удержан; детектируется точно"),
        (340, "Все категории", None, "Усть-Майский м.р. (Якутия): декабрь ×1.44 — сезонность, детектор молчит"),
        (1, "Все категории", None, "Майкоп (Адыгея): спокойный ряд — правильное молчание (0 ложных тревог)"),
    ]
    fig = plt.figure(figsize=FULL)
    axes = [fig.add_subplot(2, 2, i + 1) for i in range(4)]
    for ax, (tid, cat, true_m, title) in zip(axes, cases):
        s = panel[(panel.territory_id == tid) & (panel.category == cat)].sort_values("month_idx")
        ax.plot(s.month_idx, s.value, color=INK, lw=1.8)
        ax.set_title(title, fontsize=9.8, loc="left", color=INK, pad=4)
        if true_m is not None:
            ax.axvline(true_m, color=NEG, ls="--", lw=1.6, alpha=0.95)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(["2023-01", "2023-07", "2024-01", "2024-07", "2024-12"], fontsize=9.5)
        ax.set_ylabel("₽/жителя", fontsize=10)
        ax.tick_params(axis="y", labelsize=9.5)
        hgrid(ax)
    fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.10, hspace=0.55, wspace=0.14)
    head(fig,
         "Эталонные сдвиги уровня ловятся, сезонность Севера и спокойные ряды — нет",
         "Бордовый пунктир — истинный месяц события; в контрольных рядах (нижний ряд) "
         "события нет — правильный ответ детектора: молчание")
    foot(fig, SOURCE)
    save(fig, "F3_cp_showcase")


def f4_fan_chart(df):
    """F4: настоящий fan chart — Chronos-2 квантили (q10/q50/q90, origin 12, шаги 1..12)
    на одном показательном ряде + факт. Калибровка (coverage) — в отчёте."""
    qf = OUT / "preds_chronos2_quant_o12.parquet"
    if not qf.exists():
        print("нет квантильного файла, F4 пропущена"); return
    q = pd.read_parquet(qf)
    q = q[q.series_id.str.endswith("|Все категории")]
    # показательный ряд: медианный по уровню МО
    lvls = q.groupby("series_id")["y_true"].median().sort_values()
    sid = lvls.index[len(lvls) // 2]
    s = q[q.series_id == sid].sort_values("horizon")
    panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet")
    tid = int(sid.split("|")[0])
    hist = panel[(panel.territory_id == tid) & (panel.category == "Все категории")].sort_values("month_idx")
    hist12 = hist[hist.month_idx <= 11]
    name = panel[panel.territory_id == tid]["municipal_district_name_short"].iloc[0]
    region = panel[panel.territory_id == tid]["region_name"].iloc[0]
    coverage = ((s.y_true >= s.q10) & (s.y_true <= s.q90)).mean()

    months = pd.period_range("2023-01", periods=24, freq="M").to_timestamp()
    hx, fx = months[:12], months[12:24]

    fig = plt.figure(figsize=FULL)
    ax = fig.add_axes([0.065, 0.13, 0.90, 0.68])
    ax.fill_between(fx, s.q10, s.q90, color=OURS, alpha=0.16, lw=0)
    ax.plot(hx, hist12.value, color=INK, lw=1.9)
    ax.plot(fx, s.y_true.values, color=NEG, lw=1.9, marker="o", ms=4)
    ax.plot(fx, s.q50, color=OURS, lw=1.9)
    ax.axvline(months[11] + (months[12] - months[11]) / 2, color=MUTED, lw=1.1, ls=":")
    # прямые подписи
    ymax = max(hist12.value.max(), s.q90.max())
    ax.text(hx[6], hist12.value.iloc[6] * 0.90, "история 2023 (train)", fontsize=11,
            color=INK, ha="center", va="top")
    ax.text(fx[-1], s.y_true.iloc[-1], " факт 2024", fontsize=11, color=NEG,
            va="center", ha="left", fontweight="bold")
    ax.text(fx[-1], s.q50.iloc[-1], " медианный прогноз", fontsize=11, color=OURS,
            va="center", ha="left", fontweight="bold")
    ax.text(fx[-1], s.q90.iloc[-1], " 80% PI (Chronos-2)", fontsize=11, color=OURS,
            va="bottom", ha="left", alpha=0.85)
    ax.text(months[11] + (months[12] - months[11]) / 2, ymax * 1.005, "начало прогноза ",
            fontsize=10.5, color=MUTED, ha="right", va="bottom")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(months[0], months[-1] + pd.DateOffset(days=150))
    ax.set_ylim(0, ymax * 1.08)
    ax.set_ylabel("₽/жителя")
    hgrid(ax)
    head(fig,
         f"Факт 2024 года целиком лежит внутри 80%-го интервала Chronos-2 "
         f"({int(round(coverage * 12))} из 12 месяцев)",
         f"Ряд «{name} ({region})», «Все категории»; квантили q10/q50/q90, "
         "единый origin 2023-12, прогноз на 12 месяцев вперёд")
    foot(fig, SOURCE)
    save(fig, "F4_fan_chart")


def f1_error_map(df):
    """F1 (герой): карта RelMAE чемпиона h=3 по МО — где модель слаба (честность как сигнал).
    Проекция Albers equal-area (50°/70°N, lon₀=100°E), вся география без клипов,
    diverging-шкала с центром 1.0 (паритет с наивным), simplify 2 км topology-preserving."""
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
    ps = d[d.model == champ].groupby(d.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    sn = d[d.model == "SeasonalNaive"].groupby(d.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    rel = (ps / sn).dropna().rename("relmae").reset_index().rename(columns={"series_id": "territory_id"})
    share_bad = (rel.relmae > 1).mean()
    med = rel.relmae.median()

    g = gpd.read_file(gpkg[0])
    tid_col = [c for c in g.columns if "terr" in c.lower() or c.lower() in ("tid", "id")][0]
    g["_tid"] = g[tid_col].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    kal_mask = g.geometry.centroid.x < 25  # эксклав: 25 МО Калининградской области
    g = g.to_crs(ALBERS)
    g["geometry"] = g.simplify(2000, preserve_topology=True)  # PDF: десятки МБ → единицы
    g = g.merge(rel, left_on="_tid", right_on="territory_id", how="left")
    n_missing = int(g.relmae.isna().sum())

    fig = plt.figure(figsize=FULL)
    ax = fig.add_axes([0.005, 0.09, 0.99, 0.76])
    norm = TwoSlopeNorm(vmin=0.2, vcenter=1.0, vmax=2.0)
    have = g[g.relmae.notna()]
    have.plot(column="relmae", ax=ax, cmap=DIVERGE, norm=norm, lw=0)
    g[g.relmae.isna()].plot(ax=ax, color="#EAE4D3", edgecolor="#C7BFA9",
                            hatch="///", lw=0.1)
    g.dissolve().boundary.plot(ax=ax, color="#4C5361", lw=0.6, alpha=0.6)
    # вся география: границы от спроецированной геометрии, без перцентильных клипов
    b = g.total_bounds
    ax.set_xlim(b[0] - 8e4, b[2] + 8e4)
    ax.set_ylim(b[1] - 8e4, b[3] + 8e4)
    ax.set_axis_off()
    # Калининградский эксклав — подпись с выноской (иначе читается как неопознанный островок)
    kal = g[kal_mask.values]
    kc = kal.geometry.centroid
    kx, ky = float(kc.x.mean()), float(kc.y.mean())
    ax.annotate("Калининградская обл.", (kx, ky),
                xytext=(b[0] - 8e4 + 0.02 * (b[2] - b[0] + 1.6e5),
                        b[1] - 8e4 + 0.845 * (b[3] - b[1] + 1.6e5)),
                fontsize=10.5, color=INK, ha="left", va="center", zorder=6,
                bbox={"boxstyle": "round,pad=0.25", "fc": BG, "ec": "#D9D2BE", "lw": 0.6},
                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.7})
    # colorbar горизонтальный, с подписью метода и якорем на 1.0
    cax = fig.add_axes([0.17, 0.085, 0.42, 0.026])
    sm = plt.cm.ScalarMappable(cmap=DIVERGE, norm=norm)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal", extend="min")
    cb.set_ticks([0.2, 0.5, 1.0, 1.5, 2.0])
    cb.set_ticklabels(["0.2", "0.5", "1.0 = паритет", "1.5", "2.0"], fontsize=9.5)
    cb.set_label("RelMAE (h=3): MAE модели / MAE сезонного наивного на том же ряду",
                 fontsize=10.5, color=INK)
    cb.ax.xaxis.set_label_position("top")
    cb.outline.set_visible(False)
    fig.legend(handles=[Patch(facecolor="#EAE4D3", edgecolor="#C7BFA9", hatch="///",
                              label=f"вне панели: {n_missing} МО (нет ряда в данных)")],
               loc="lower right", bbox_to_anchor=(0.995, 0.035), fontsize=10, frameon=False)
    head(fig,
         f"{champ} хуже сезонного наивного лишь в {share_bad * 100:.1f}% "
         f"из {len(rel):_} муниципалитетов".replace("_", " "),
         f"RelMAE чемпиона, h=3, «Все категории»; медиана по стране {med:.2f} — "
         f"на {(1 - med) * 100:.0f}% точнее наивного; тёмно-бордовое — где модель слаба")
    foot(fig, SOURCE + " Полигоны МО: t_dict_municipal_districts_poly.gpkg; "
                       "проекция Albers equal-area (50°/70°N, центральный меридиан 100°E).")
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
