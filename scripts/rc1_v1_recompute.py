"""rc1 V1: независимый пересчёт хедлайн-чисел methodology.md из сырых outputs/data.

Ничего не перезаписывает из существующих outputs; пишет только outputs/rc1_v1_*.csv.
Пункты 1-10 брифа. Запуск: .venv/bin/python scripts/rc1_v1_recompute.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# ровно как scripts/04_build_tables.py
EXCLUDE_GLOB = ["preds_stats_ces_part*", "preds_stats_steps_part*",
                "preds_chronos2_quant_o12*", "preds_chronos2_zs_raw_o12*",
                "preds_coldstart*"]
HORIZONS = [1, 3, 6, 12]
results = []


def rec(item, where, claimed, recomputed, verdict, note=""):
    results.append({"item": item, "where": where, "claimed": claimed,
                    "recomputed": recomputed, "verdict": verdict, "note": note})
    print(f"[{verdict:8s}] {item}: claimed={claimed} recomputed={recomputed} {note}")


# ---------- пп.1-2: matched cells + RelMAE из preds ----------
def load_all():
    parts = []
    for f in sorted(OUT.glob("preds_*.parquet")):
        if any(f.match(p) for p in EXCLUDE_GLOB):
            continue
        df = pd.read_parquet(f)
        if "model" not in df.columns:
            df["model"] = f.stem.replace("preds_", "")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = df[df["model"] != "y"]
    return df.drop_duplicates(["series_id", "origin", "horizon", "model"], keep="last")


def matched_cells(df):
    keep = []
    for h, g in df.groupby("horizon"):
        sets = [set(zip(gg["series_id"], gg["origin"])) for _, gg in g.groupby("model")]
        inter = set.intersection(*sets) if sets else set()
        sel = g[g.apply(lambda r: (r["series_id"], r["origin"]) in inter, axis=1)]
        keep.append(sel)
    return pd.concat(keep, ignore_index=True)


def relmae(df, model, h, snaive):
    """Σ per-series MAE(model) / Σ per-series MAE(SNaive) на тех же ячейках."""
    a = df[(df.model == model) & (df.horizon == h)].copy()
    a["ae"] = (a.y_true - a.y_pred).abs()
    b = snaive[snaive.horizon == h].copy()
    b["ae_sn"] = (b.y_true - b.y_pred).abs()
    key = ["series_id", "origin"]
    m = a.merge(b[key + ["ae_sn"]], on=key, how="inner")
    ps = m.groupby("series_id").agg(ae=("ae", "mean"), ae_sn=("ae_sn", "mean"))
    return ps.ae.sum() / ps.ae_sn.sum(), ps


print("=== load preds ===", flush=True)
df = load_all()
print(f"rows={len(df)}, models={sorted(df.model.unique())}")
dm = matched_cells(df)
print(f"matched rows={len(dm)}")
dm["category"] = dm.series_id.str.split("|").str[1]
snaive = dm[dm.model == "SeasonalNaive"][["series_id", "origin", "horizon", "y_true", "y_pred"]]

rows_t1, rows_cat = [], {}
for h in HORIZONS:
    for mdl in ["Prophet-B*", "DriftAdjSNaive", "SeasProfileHA", "Prophet-A"]:
        v, ps = relmae(dm, mdl, h, snaive)
        rows_t1.append({"model": mdl, "horizon": h, "relmae_pooled": v,
                        "n_series": len(ps)})
        sub = dm[dm.category == "Все категории"]
        sn_sub = snaive[snaive.series_id.isin(sub.series_id.unique())]
        vc, psc = relmae(sub, mdl, h, sn_sub)
        rows_cat.setdefault(mdl, {})[h] = (vc, len(psc))
t1 = pd.DataFrame(rows_t1)
t1.to_csv(OUT / "rc1_v1_relmae_recompute.csv", index=False)

pb = {r.horizon: r.relmae_pooled for r in t1.itertuples() if r.model == "Prophet-B*"}
rec("P2 Prophet-B* pooled RelMAE h=1/3/6/12", "T1/отчёт табл.T1",
    "0.457/0.412/0.521/0.626",
    "/".join(f"{pb[h]:.3f}" for h in HORIZONS),
    "MATCH" if all(abs(pb[h] - c) < 0.0005 for h, c in
                   zip(HORIZONS, [0.457, 0.412, 0.521, 0.626])) else "MISMATCH")

das = rows_cat["DriftAdjSNaive"]
rec("P1 DriftAdjSNaive «Все категории» RelMAE h=1/3/6", "отчёт табл. «Все категории»",
    "0.324/0.361/0.422",
    "/".join(f"{das[h][0]:.3f}" for h in [1, 3, 6]),
    "MATCH" if all(abs(das[h][0] - c) < 0.0005 for h, c in
                   zip([1, 3, 6], [0.324, 0.361, 0.422])) else "MISMATCH",
    f"n_series={das[1][1]}")
sp = rows_cat["SeasProfileHA"]
rec("P1 SeasProfileHA «Все категории» RelMAE h=12", "отчёт табл. «Все категории»",
    "0.513", f"{sp[12][0]:.3f}",
    "MATCH" if abs(sp[12][0] - 0.513) < 0.0005 else "MISMATCH", f"n_series={sp[12][1]}")

# Prophet-B* на срезе «Все категории» (контрольные 0.460/0.398/0.494/0.617)
pbc = rows_cat["Prophet-B*"]
rec("P2b Prophet-B* «Все категории» h=1/3/6/12", "отчёт табл. «Все категории»",
    "0.460/0.398/0.494/0.617", "/".join(f"{pbc[h][0]:.3f}" for h in HORIZONS),
    "MATCH" if all(abs(pbc[h][0] - c) < 0.0005 for h, c in
                   zip(HORIZONS, [0.460, 0.398, 0.494, 0.617])) else "MISMATCH")

# ---------- п.3: парный Wilcoxon h=1, «Все категории», DriftAdjSNaive vs Prophet-B* ----------
sub = dm[(dm.category == "Все категории")]
_, ps_a = relmae(sub, "DriftAdjSNaive", 1, snaive)
_, ps_b = relmae(sub, "Prophet-B*", 1, snaive)
common = ps_a.index.intersection(ps_b.index)
d = (ps_a.loc[common, "ae"] - ps_b.loc[common, "ae"]).dropna()
d = d[d != 0]
w_raw = stats.wilcoxon(d).pvalue
w_holm4 = min(w_raw * 4, 1.0)  # семейство из 4 headline-тестов (24_t5)
rec("P3 Wilcoxon h=1 DriftAdjSNaive vs Prophet-B* (Все категории)",
    "отчёт табл. «Все категории», колонка Wilcoxon p (Holm)",
    "1.7e-241", f"raw={w_raw:.2e}, Holm×4={w_holm4:.2e}",
    "MATCH" if abs(np.log10(w_raw) - np.log10(1.7e-241)) < 0.6
    or abs(np.log10(w_holm4) - np.log10(1.7e-241)) < 0.6 else "MISMATCH",
    f"n={len(d)}, median_diff={d.median():.1f}; pipeline T5 w_p=3.43e-242")

# ---------- п.4: форвард-2025 ----------
chk = pd.read_csv(OUT / "forecast_2025_check.csv")
def fwd_from_check(scope):
    s = chk[(chk.scope == scope) & (chk.fc_model == "DriftAdjSNaive")].copy()
    s["ae"] = (s.fc_yoy_pct - s.official_yoy_pct).abs()
    s["in80"] = (s.official_yoy_pct >= s.fc_yoy_lo80_pct) & (s.official_yoy_pct <= s.fc_yoy_hi80_pct)
    return s
for scope, claim in [("Все категории", 1.48), ("Продовольствие", 1.36)]:
    s = fwd_from_check(scope)
    rec(f"P4 форвард-2025 MAE {scope} (из forecast_2025_check.csv)", "отчёт строка К7",
        f"{claim}", f"{s.ae.mean():.2f} (n={len(s)}, in80={s.in80.sum()}/{len(s)})",
        "MATCH" if abs(s.ae.mean() - claim) < 0.01 else "MISMATCH")

# независимо из parquet: сумма прогноза по МО / факт панели год назад
panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet",
                        columns=["territory_id", "category", "date", "value"])
fc = pd.read_parquet(OUT / "forecast_2025H1.parquet")
act = {c: panel[panel.category == c].groupby("date")["value"].sum() for c in
       ["Все категории", "Продовольствие"]}
rows4 = []
for scope, claim in [("Все категории", 1.48), ("Продовольствие", 1.36)]:
    sub = fc[(fc.category == scope) & (fc.model == "DriftAdjSNaive")]
    g = sub.groupby("month").agg(pred=("y_pred", "sum"), lo=("y_lo80", "sum"),
                                 hi=("y_hi80", "sum"))
    a = act[scope]
    off = chk[(chk.scope == scope) & (chk.fc_model == "DriftAdjSNaive")].set_index("month")["official_yoy_pct"]
    ae, n_in = [], 0
    for m in g.index:
        prev = a.get(str(pd.Period(m, freq="M") - 12))
        if prev is None or m not in off.index:
            continue
        yoy = (g.loc[m, "pred"] / prev - 1) * 100
        lo = (g.loc[m, "lo"] / prev - 1) * 100
        hi = (g.loc[m, "hi"] / prev - 1) * 100
        ae.append(abs(yoy - off[m]))
        n_in += int(lo <= off[m] <= hi)
        rows4.append({"scope": scope, "month": m, "fc_yoy": yoy, "official": off[m],
                      "lo80": lo, "hi80": hi})
    rec(f"P4 форвард-2025 MAE {scope} (пересчёт из forecast_2025H1.parquet + panel)",
        "отчёт строка К7", f"{claim}, 6/6 в PI80",
        f"MAE={np.mean(ae):.2f}, in80={n_in}/{len(ae)}",
        "MATCH" if abs(np.mean(ae) - claim) < 0.01 and n_in == 6 else "MISMATCH")
pd.DataFrame(rows4).to_csv(OUT / "rc1_v1_forward2025_recompute.csv", index=False)

# ---------- п.5: внешняя валидация r ----------
allcat = panel[panel.category == "Все категории"]
agg = allcat.groupby("date")["value"].agg(panel_sum="sum", panel_median="median")
growth = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" /
                         "consumer-spending-growth_parquet.parquet")
levels = pd.read_parquet(ROOT / "data" / "external" / "sberindex_api" /
                         "consumer-spending_parquet.parquet")
off_lvl = levels[levels["type"] == "Всего"].set_index("period")["value"].sort_index()
off_lvl.index = pd.PeriodIndex(off_lvl.index.astype(str), freq="M")
agg.index = pd.PeriodIndex(agg.index.astype(str), freq="M")
lvl = pd.DataFrame({"panel_sum": agg.panel_sum, "official": off_lvl}).dropna()
r_lvl = lvl.panel_sum.corr(lvl.official)
r_mom = np.log(lvl.panel_sum).diff().corr(np.log(lvl.official).diff())
rec("P5 r уровни (24 мес)", "отчёт Находка 4", "0.955", f"{r_lvl:.3f} (n={len(lvl)})",
    "MATCH" if abs(r_lvl - 0.955) < 0.005 else "MISMATCH")
rec("P5 r м/м log-приросты", "отчёт Находка 4", "0.922", f"{r_mom:.3f}",
    "MATCH" if abs(r_mom - 0.922) < 0.005 else "MISMATCH",
    "NB: в отчёте «м/м приросты», в коде — log-diff")

# ---------- п.6: покрытие интервалов из intervals_champion.parquet ----------
iv = pd.read_parquet(OUT / "intervals_champion.parquet")
wide_rub = panel.copy()
wide_rub["month_idx"] = (pd.PeriodIndex(wide_rub.date, freq="M").year - 2023) * 12 + \
                        pd.PeriodIndex(wide_rub.date, freq="M").month - 1
wide_rub["series_id"] = wide_rub.territory_id.astype(str) + "|" + wide_rub.category
truth = wide_rub.set_index(["series_id", "month_idx"])["value"].rename("y_true")
iv["tm"] = iv.origin + iv.horizon - 1
iv = iv.join(truth, on=["series_id", "tm"])
iv["cov80"] = (iv.y_true >= iv.lo80) & (iv.y_true <= iv.hi80)
rows6 = []
for h in [1, 3, 6]:
    s = iv[(iv.model == "DriftAdjSNaive") & (iv.horizon == h)]
    s1 = s[s.tier == 1]
    rows6.append({"slice": f"h{h}_tier1", "cov80": s1.cov80.mean(), "n": len(s1)})
    rec(f"P6 coverage 80PI DriftAdjSNaive h={h} (tier1)", "отчёт К7 интервалы",
        f"{[0.860, 0.903, 0.927][[1, 3, 6].index(h)]}", f"{s1.cov80.mean():.3f}",
        "MATCH" if abs(s1.cov80.mean() - [0.860, 0.903, 0.927][[1, 3, 6].index(h)]) < 0.001
        else "MISMATCH")
# малые МО Q1 h=1
scale = wide_rub.groupby("series_id")["value"].median()
q = pd.qcut(scale, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
iv["scale_q"] = iv.series_id.map(q).astype(str)
s = iv[(iv.model == "DriftAdjSNaive") & (iv.horizon == 1) & (iv.scale_q == "Q1")]
rec("P6 coverage малые МО (Q1, h=1)", "отчёт К7", "0.734", f"{s.cov80.mean():.3f} (n={len(s)})",
    "MATCH" if abs(s.cov80.mean() - 0.734) < 0.001 else "MISMATCH")
# декабрь-2024 (tm=23), чемпион-of-record
CH = {1: "DriftAdjSNaive", 3: "DriftAdjSNaive", 6: "DriftAdjSNaive", 12: "SeasProfileHA"}
chd = pd.concat([iv[(iv.model == m) & (iv.horizon == h)] for h, m in CH.items()])
s = chd[chd.tm == 23]
rec("P6 coverage декабрь-2024 (champion-of-record, tm=23)", "отчёт К7",
    "0.821", f"{s.cov80.mean():.3f} (n={len(s)})",
    "MATCH" if abs(s.cov80.mean() - 0.821) < 0.001 else "MISMATCH")
pd.DataFrame(rows6).to_csv(OUT / "rc1_v1_coverage_recompute.csv", index=False)

# ---------- п.7: LGBM сиды ----------
seeds = pd.read_csv(OUT / "lgbm_seeds.csv")
print(seeds.to_string())
piv = seeds.pivot_table(index="seed", columns="horizon", values="relmae_vs_snaive") \
    if "seed" in seeds.columns else None
if piv is None:
    piv = seeds.set_index(seeds.columns[0])
    piv.columns = piv.columns.astype(str)
for h, claim_m, claim_s in [(1, 0.481, 0.009), (3, 0.548, 0.009), (6, 0.690, 0.017)]:
    col = h if h in piv.columns else str(h)
    v = piv[col].astype(float)
    rec(f"P7 LGBM 5 сидов h={h}", "отчёт appendix", f"{claim_m}±{claim_s}",
        f"{v.mean():.3f}±{v.std(ddof=1):.3f} (n={len(v)})",
        "MATCH" if abs(v.mean() - claim_m) < 0.002 and abs(v.std(ddof=1) - claim_s) < 0.003
        else "MISMATCH")

# ---------- п.8: экономика ----------
a23 = agg.panel_sum.loc["2023-01":"2023-12"].sum()
a24 = agg.panel_sum.loc["2024-01":"2024-12"].sum()
nom_sum = (a24 / a23 - 1) * 100
med_row = agg.panel_median
nom_med_row = (med_row.loc["2024-01":"2024-12"].sum() / med_row.loc["2023-01":"2023-12"].sum() - 1) * 100
# per-МО YoY 2024/2023 (Все категории): медиана и среднее
per_mo = allcat.pivot_table(index="territory_id", columns="date", values="value")
y23 = per_mo[[c for c in per_mo.columns if str(c) <= "2023-12"]].sum(axis=1)
y24 = per_mo[[c for c in per_mo.columns if str(c) >= "2024-01"]].sum(axis=1)
per_yoy = (y24 / y23 - 1) * 100
cpi = pd.read_csv(ROOT / "data" / "external" / "cbr_cpi_yoy_2023_2024.csv", sep=";")
cpi["cpi_yoy"] = cpi.cpi_yoy.astype(str).str.replace(" ", "").str.replace(",", ".").astype(float)
cpi["date"] = pd.PeriodIndex(pd.to_datetime(cpi.month.astype(str).str.strip(),
                                            format="%m.%Y"), freq="M")
cpi = cpi.set_index("date").cpi_yoy.sort_index()
cpi24 = cpi.loc["2024-01":"2024-12"].mean()
real_sum = ((1 + nom_sum / 100) / (1 + cpi24 / 100) - 1) * 100
real_med = ((1 + nom_med_row / 100) / (1 + cpi24 / 100) - 1) * 100
# помесячный реальный YoY панели (сумма) — среднее за 2024
nom_m = agg.panel_sum.pct_change(12) * 100
real_m = ((1 + nom_m / 100) / (1 + cpi.reindex(nom_m.index) / 100) - 1) * 100
real_m24 = real_m.loc["2024-01":"2024-12"]
rec("P8 номинал 2024/2023: медиана по МО / среднее по МО", "отчёт Находка 4",
    "+15.1% / +14.8%", f"{per_yoy.median():.1f}% / {per_yoy.mean():.1f}%",
    "MATCH" if abs(per_yoy.median() - 15.1) < 0.05 and abs(per_yoy.mean() - 14.8) < 0.06
    else "MISMATCH",
    f"для сверки: сумма {nom_sum:.2f}%, медианный ряд {nom_med_row:.2f}%")
rec("P8 реальный рост за год (ИПЦ ЦБ)", "отчёт Находка 4", "+6.2%",
    f"sum-вариант {real_sum:.2f}% | med-row {real_med:.2f}% | mean-monthly {real_m24.mean():.2f}%",
    "MATCH" if min(abs(real_sum - 6.2), abs(real_med - 6.2), abs(real_m24.mean() - 6.2)) < 0.06
    else "MISMATCH",
    f"ИПЦ ср.2024={cpi24:.2f}%")

# ---------- п.9: CPD ----------
fr = pd.read_csv(OUT / "cpd_frontier_v2.csv")
b = fr[(fr.method == "BOCPD") & (fr.far_per_1000 <= 2)].sort_values("far_per_1000")
best = b.loc[b.recall.idxmax()]
x, n = round(best.recall * 378), 378
lo = stats.beta.ppf(0.025, x, n - x + 1) * 100
hi = stats.beta.ppf(0.975, x + 1, n - x) * 100
rec("P9 BOCPD recall @ FAR≤2/1000 (1.6σ)", "отчёт F3.1", "5.6% [3.8–7.6]",
    f"{best.recall*100:.2f}% (FAR={best.far_per_1000:.2f}, delta={best.delta}); "
    f"CP-CI 21/378=[{lo:.1f}–{hi:.1f}]",
    "MATCH" if abs(best.recall - 0.056) < 0.001 else "MISMATCH",
    "CI [3.8–7.6] НЕ воспроизводится ни одним стандартным методом (CP 3.5–8.4, Wilson 3.7–8.3)")
mr = pd.read_csv(OUT / "cpd_matched_rate_v3.csv")
f1b = mr[mr.method == "BOCPD"].F1.max()
f1w = mr[mr.method == "Window"].F1.max()
rec("P9 matched F1: BOCPD / Window", "отчёт F3.1а", "0.67 / 0.64",
    f"{f1b:.3f} / {f1w:.3f}",
    "MATCH" if abs(f1b - 0.67) < 0.005 and abs(f1w - 0.64) < 0.005 else "MISMATCH")
rec("P9 инъекций", "отчёт F3.1", "378", f"n_series={mr.n_series.unique()}",
    "MATCH" if list(mr.n_series.unique()) == [378] else "MISMATCH")
sp9 = pd.read_csv(OUT / "spatial_propagation.csv")
rec("P9 соседские пары ≤80км", "отчёт F3.3", "42404",
    f"{sp9.pairs_le80km.unique()} (файл: {len(sp9)} метод-строк)",
    "MATCH" if list(sp9.pairs_le80km.unique()) == [42404] else "MISMATCH")

# ---------- п.10: новости ----------
abl = pd.read_csv(OUT / "news_ablation_N0N3.csv")
n1 = abl[(abl.config == "N1") & (abl.vs == "N0") & (abl.h == 1)].delta_pct.iloc[0]
n2 = abl[(abl.config == "N2") & (abl.vs == "N0") & (abl.h == 1)].delta_pct.iloc[0]
n3 = abl[(abl.config == "N3") & (abl.vs == "N0") & (abl.h == 1)].delta_pct.iloc[0]
rec("P10 новости ΔMAE h=1 N1/N2/N3 vs N0", "отчёт §новости",
    "+10.8/+15.9/+20.7%", f"+{n1:.1f}/+{n2:.1f}/+{n3:.1f}%",
    "MATCH" if abs(n1 - 10.8) < 0.05 and abs(n2 - 15.9) < 0.05 and abs(n3 - 20.7) < 0.05
    else "MISMATCH")

# κ: независимый пересчёт из выборки — заголовки CSV + таксономия yaml + human-разметка
# (human-дикт скопирован дословно из scripts/17_news_kappa.py — это единственный источник разметки)
import re
import yaml

TOPICS = ["prices_inflation", "fx_ruble", "rate_credit",
          "sanctions_shocks", "consumer_sentiment", "emergencies_regional"]
SHORT = {"prices_inflation": "prices", "fx_ruble": "fx", "rate_credit": "rate",
         "sanctions_shocks": "sanctions", "consumer_sentiment": "sentiment",
         "emergencies_regional": "emerg"}
HUMAN = {
    0: "", 1: "", 2: "prices,rate,fx", 3: "sanctions", 4: "emerg", 5: "sanctions",
    6: "prices", 7: "", 8: "prices", 9: "sanctions", 10: "emerg", 11: "prices",
    12: "sentiment", 13: "prices", 14: "sanctions", 15: "prices", 16: "", 17: "fx",
    18: "", 19: "emerg", 20: "", 21: "fx", 22: "emerg", 23: "rate", 24: "",
    25: "emerg", 26: "", 27: "sanctions", 28: "sanctions", 29: "sanctions",
    30: "prices", 31: "rate", 32: "rate", 33: "", 34: "", 35: "rate", 36: "rate",
    37: "prices", 38: "", 39: "emerg", 40: "emerg", 41: "sanctions", 42: "",
    43: "rate", 44: "rate", 45: "sanctions", 46: "", 47: "", 48: "rate", 49: "fx",
}


def stem_regex(kw):
    return r"\s+".join(re.escape(tok[:-1]) + r"\w*" if tok.endswith("*")
                       else re.escape(tok) + r"\w*" for tok in kw.lower().split())


tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
geo_excl = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_exclude"]) + r")")
geo_resc = re.compile(r"(?:^|\W)(" + "|".join(stem_regex(s) for s in tax["geo_rescue"]) + r")")
kw_pats = {t: re.compile(r"(?:^|\W)(" + "|".join(stem_regex(k) for k in spec["keywords"]) + r")")
           for t, spec in tax["topics"].items()}
ks = pd.read_csv(OUT / "news_kappa_sample_v2.csv")
ks["t"] = ks.title.str.lower()
is_f = ks["t"].str.contains(geo_excl)
has_r = ks["t"].str.contains(geo_resc)
kappas, conf = {}, {}
for t in TOPICS:
    spec = tax["topics"][t]
    raw = ks["t"].str.contains(kw_pats[t])
    if spec.get("status") == "retired":
        d = pd.Series(False, index=ks.index)
    elif spec.get("geo") == "with_rescue":
        d = raw & ~(is_f & ~has_r)
    else:
        d = raw & ~is_f
    h = np.array([SHORT[t] in (HUMAN[i].split(",") if HUMAN[i] else []) for i in ks["id"]])
    d = d.to_numpy()
    tp = int((d & h).sum()); fp = int((d & ~h).sum()); fn = int((~d & h).sum()); tn = int((~d & ~h).sum())
    po = (tp + tn) / 50
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / 50 ** 2
    kappas[t] = (po - pe) / (1 - pe)
    conf[t] = (tp, fp, fn, tn)
active = [t for t in TOPICS if tax["topics"][t].get("status") != "retired"]
macro = float(np.mean([kappas[t] for t in active]))
rec("P10 macro-κ (активные темы)", "отчёт §новости / news_kappa_v2.md", "0.926 (n=50)",
    f"{macro:.4f}; per-theme: " +
    "; ".join(f"{t}={kappas[t]:.3f}{conf[t]}" for t in TOPICS),
    "MATCH" if abs(macro - 0.926) < 0.0005 else "MISMATCH",
    f"n={len(ks)}; пересчёт из заголовков × таксономия × human-разметка")

pd.DataFrame(results).to_csv(OUT / "rc1_v1_summary.csv", index=False)
print("\nSaved outputs/rc1_v1_summary.csv")
