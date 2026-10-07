"""rc1 S4: исправленная F1 — карта ошибок чемпиона (h=3, «Все категории»).

Проблемы оригинала (reports/figures/F1_error_map.png):
- в грейскейле ярко-жёлтый high-end cividis (RelMAE≈2) сливается со светло-серым
  «вне панели» (#dddddd) — худшие МО визуально принимаются за отсутствие данных;
- подпись colorbar мелкая для слайда; заголовок-тема, а не вывод;
- PDF-версия весит 34 МБ (векторные полигоны) — здесь только растр 200 dpi.

Фикс: cividis_r (плохое = ТЁМНОЕ, в ч/б однозначно отделено от светло-серого
missing), missing со штриховкой, крупный colorbar, заголовок-вывод
(доля 3,8% посчитана из outputs, не с потолка).
Данные и расчёт rel — дословно как в scripts/17_figures.py (f1_error_map).
Запуск: .venv/bin/python scripts/rc1_s4_fix_f1.py
Выход: outputs/rc1_s4_F1_error_map.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

plt.rcParams.update({
    "font.family": "PT Sans", "axes.unicode_minus": False,
    "font.size": 12.5, "axes.titlesize": 15,
})


def main():
    import geopandas as gpd
    gpkg = list((ROOT / "data" / "raw").rglob("*.gpkg"))
    assert gpkg, "нет полигонов gpkg"

    files = [f for f in OUT.glob("preds_*.parquet")
             if not f.match("*_ces_part*") and not f.match("*_steps_part*")
             and not f.match("*_quant_o12*") and not f.match("*_raw_o12*")
             and not f.match("*coldstart*")]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df[df["model"] != "y"].drop_duplicates(["series_id", "origin", "horizon", "model"])

    t1 = pd.read_csv(OUT / "T1_model_x_horizon.csv")
    champ = t1[t1.horizon == 3].sort_values("relmae_vs_snaive").iloc[0]["model"]

    d = df[(df.horizon == 3) & (df.series_id.str.endswith("|Все категории"))].copy()
    d["ae"] = (d.y_true - d.y_pred).abs()
    ps = d[d.model == champ].groupby(
        d.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    sn = d[d.model == "SeasonalNaive"].groupby(
        d.series_id.str.split("|").str[0].astype(int))["ae"].mean()
    rel = (ps / sn).dropna().rename("relmae").reset_index().rename(
        columns={"series_id": "territory_id"})
    share_bad = (rel.relmae > 1).mean()

    g = gpd.read_file(gpkg[0])
    tid_col = [c for c in g.columns if "terr" in c.lower() or c.lower() in ("tid", "id")][0]
    g["_tid"] = g[tid_col].astype(str).str.replace(r"\.0$", "", regex=True).astype(int)
    g = g.merge(rel, left_on="_tid", right_on="territory_id", how="left")

    fig, ax = plt.subplots(figsize=(12, 6.4))
    g.plot(column="relmae", ax=ax, cmap="cividis_r", vmin=0.3, vmax=2.0,
           missing_kwds={"color": "#f2f2f2", "edgecolor": "#bbbbbb",
                         "hatch": "///", "lw": 0.2},
           legend=True,
           legend_kwds={"label": "RelMAE чемпиона против seasonal naive (h=3)",
                        "shrink": 0.85})
    # однородное тело карты без антимеридианного фрагмента Чукотки
    b = g.total_bounds
    xs = g.geometry.centroid.x
    ax.set_xlim(np.nanpercentile(xs, 0.5) - 2, np.nanpercentile(xs, 99.9) + 2)
    ax.set_ylim(b[1] - 1, b[3] + 1)
    ax.set_axis_off()
    cax = fig.axes[-1]
    cax.tick_params(labelsize=12)
    cax.yaxis.label.set_size(13)
    ax.set_title(f"{champ} хуже сезонного наивного лишь в {share_bad*100:.1f}% МО "
                 f"(тёмное: Север и Дальний Восток),\nв медиане по стране ошибка "
                 f"на {(1 - rel.relmae.median())*100:.0f}% ниже наивного")
    fig.tight_layout()
    out = OUT / "rc1_s4_F1_error_map.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out, "| champ =", champ,
          "| доля rel>1 = %.3f" % share_bad,
          "| медиана = %.3f" % rel.relmae.median())


if __name__ == "__main__":
    main()
