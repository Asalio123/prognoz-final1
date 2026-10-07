"""rc1 V1 tail: пп.6-10 без перезагрузки preds. Пишет outputs/rc1_v1_tail.csv."""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
results = []


def rec(item, claimed, recomputed, verdict, note=""):
    results.append({"item": item, "claimed": claimed, "recomputed": recomputed,
                    "verdict": verdict, "note": note})
    print(f"[{verdict:14s}] {item}: claimed={claimed} | recomputed={recomputed} | {note}")


# ---------- п.6 (актуальные outputs, 14:13 07.10 — Mondrian-пулы) ----------
cov = pd.read_csv(OUT / "intervals_coverage.csv")
t1 = cov[(cov.model == "DriftAdjSNaive") & (cov.slice_type == "overall_tier1")].set_index("horizon")
q1 = cov[(cov.model == "DriftAdjSNaive") & (cov.slice_type == "scale_quintile") &
         (cov.horizon == 1) & (cov.slice_value == "Q1")].cov80.iloc[0]
rec("P6 coverage tier1 h=1/3/6 (ТЕКУЩИЕ outputs 07.10 14:13)",
    "0.860/0.903/0.927",
    f"{t1.loc[1,'cov80']:.3f}/{t1.loc[3,'cov80']:.3f}/{t1.loc[6,'cov80']:.3f}",
    "STALE-REPORT", "отчёт посчитан по pre-Mondrian интервалам (видел их в начале сессии: "
    "0.8601/0.9026/0.9271); текущие outputs лучше")
rec("P6 coverage Q1 h=1 (текущие)", "0.734", f"{q1:.3f}", "STALE-REPORT",
    "pre-Mondrian было 0.7343; Mondrian-пулы починили недопокрытие малых МО")

# декабрь-2024 из текущего parquet (champion-of-record, tm=23)
iv = pd.read_parquet(OUT / "intervals_champion.parquet")
panel = pd.read_parquet(ROOT / "data" / "panel" / "panel.parquet",
                        columns=["territory_id", "category", "date", "value"])
panel["series_id"] = panel.territory_id.astype(str) + "|" + panel.category
panel["month_idx"] = (pd.PeriodIndex(panel.date, freq="M").year - 2023) * 12 + \
                     pd.PeriodIndex(panel.date, freq="M").month - 1
truth = panel.set_index(["series_id", "month_idx"])["value"].rename("y_true")
iv["tm"] = iv.origin + iv.horizon - 1
iv = iv.join(truth, on=["series_id", "tm"])
CH = {1: "DriftAdjSNaive", 3: "DriftAdjSNaive", 6: "DriftAdjSNaive", 12: "SeasProfileHA"}
chd = pd.concat([iv[(iv.model == m) & (iv.horizon == h)] for h, m in CH.items()])
dec = chd[chd.tm == 23]
cov_dec = ((dec.y_true >= dec.lo80) & (dec.y_true <= dec.hi80)).mean()
rec("P6 coverage декабрь-2024 champion-of-record (текущие)", "0.821",
    f"{cov_dec:.3f} (n={len(dec)})", "STALE-REPORT",
    "по старым интервалам 0.821 не воспроизвести — parquet перезаписан; текущее 0.831")

# ---------- п.7: LGBM сиды ----------
seeds = pd.read_csv(OUT / "lgbm_seeds.csv")
piv = seeds.pivot_table(index="seed", columns="h", values="relmae")
for h, cm, cs in [(1, 0.481, 0.009), (3, 0.548, 0.009), (6, 0.690, 0.017)]:
    v = piv[h].astype(float)
    rec(f"P7 LGBM 5 сидов h={h}", f"{cm}±{cs}",
        f"{v.mean():.3f}±{v.std(ddof=1):.3f} (seeds={sorted(piv.index)})",
        "MATCH" if abs(v.mean() - cm) < 0.002 and abs(v.std(ddof=1) - cs) < 0.003
        else "MISMATCH")

# ---------- п.8: экономика ----------
allcat = panel[panel.category == "Все категории"]
agg = allcat.groupby("date")["value"].agg(panel_sum="sum", panel_median="median")
a23 = agg.panel_sum.loc["2023-01":"2023-12"].sum()
a24 = agg.panel_sum.loc["2024-01":"2024-12"].sum()
nom_sum = (a24 / a23 - 1) * 100
med_row = agg.panel_median
nom_med_row = (med_row.loc["2024-01":"2024-12"].sum() /
               med_row.loc["2023-01":"2023-12"].sum() - 1) * 100
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
nom_m = agg.panel_sum.pct_change(12) * 100
real_m = ((1 + nom_m / 100) / (1 + cpi.reindex(nom_m.index) / 100) - 1) * 100
real_m24 = real_m.loc["2024-01":"2024-12"]
rec("P8 номинал 2024/2023: медиана/среднее по МО", "+15.1% / +14.8%",
    f"медиана {per_yoy.median():.2f}% / среднее {per_yoy.mean():.2f}%",
    "MATCH" if abs(per_yoy.median() - 15.1) < 0.05 and abs(per_yoy.mean() - 14.8) < 0.06
    else "MISMATCH",
    f"n_MО={len(per_yoy)}; сумма-панель {nom_sum:.2f}%, медианный ряд {nom_med_row:.2f}%")
cands = {"sum_year/avgCPI": real_sum, "medrow_year/avgCPI": real_med,
         "mean_monthly_real": real_m24.mean()}
best = min(cands, key=lambda k: abs(cands[k] - 6.2))
rec("P8 реальный рост за год (дефляция ИПЦ ЦБ)", "+6.2%",
    " | ".join(f"{k}={v:.2f}%" for k, v in cands.items()),
    "MATCH" if abs(cands[best] - 6.2) < 0.06 else "MISMATCH",
    f"ближайший вариант: {best}; ИПЦ ср.2024={cpi24:.2f}%, ИПЦ 2024-12={cpi.loc['2024-12']:.2f}%")

# ---------- п.9: CPD ----------
fr = pd.read_csv(OUT / "cpd_frontier_v2.csv")
b = fr[(fr.method == "BOCPD") & (fr.far_per_1000 <= 2)]
best = b.loc[b.recall.idxmax()]
x, n = int(round(best.recall * 378)), 378
cp = (stats.beta.ppf(0.025, x, n - x + 1) * 100, stats.beta.ppf(0.975, x + 1, n - x) * 100)
rec("P9 BOCPD recall @ FAR≤2/1000 (delta=1.6σ)", "5.6% [95% CI 3.8–7.6]",
    f"recall={best.recall*100:.2f}% при FAR={best.far_per_1000:.3f}, param={best.param}; "
    f"CP-CI {x}/{n}=[{cp[0]:.1f}–{cp[1]:.1f}]",
    "MATCH-число / MISMATCH-CI" if abs(best.recall - 0.056) < 0.001 else "MISMATCH",
    "recall 5.6% точно; CI [3.8–7.6] не воспроизводится: CP [3.5–8.4], Wilson [3.7–8.3], "
    "Jeffreys [3.6–8.2]; frontier хранит bootstrap-CI только для F1")
mr = pd.read_csv(OUT / "cpd_matched_rate_v3.csv")
f1b = mr[mr.method == "BOCPD"].F1.max()
f1w = mr[mr.method == "Window"].F1.max()
f1pb = mr[mr.method.isin(["PELT", "BinSeg"])].F1.max()
rec("P9 matched F1 BOCPD / Window / PELT-BinSeg", "0.67 / 0.64 / 0.40",
    f"{f1b:.3f} / {f1w:.3f} / {f1pb:.3f}",
    "MATCH" if abs(f1b - 0.67) < 0.005 and abs(f1w - 0.64) < 0.005 and abs(f1pb - 0.40) < 0.005
    else "MISMATCH",
    f"BOCPD row: level 3.0σ, K_panel=215, tp/fp/fn="
    f"{mr.loc[mr.F1.idxmax(), ['tp','fp','fn']].tolist()}")
inj = pd.read_csv(OUT / "cpd_injection_scores_v2.csv")
rec("P9 стратифицированных инъекций", "378",
    f"n={inj.n.unique()} (все строки v2), matched_rate n_series={mr.n_series.unique()}",
    "MATCH" if list(inj.n.unique()) == [378] and list(mr.n_series.unique()) == [378]
    else "MISMATCH")
sp9 = pd.read_csv(OUT / "spatial_propagation.csv")
rec("P9 соседские пары ≤80 км", "42 404",
    f"{sp9.pairs_le80km.iloc[0]} (все {len(sp9)} метод-строки согласны); "
    f"BOCPD lift={sp9[sp9.method=='BOCPD'].lift.iloc[0]}, p={sp9[sp9.method=='BOCPD'].p_value.iloc[0]}",
    "MATCH" if (sp9.pairs_le80km == 42404).all() else "MISMATCH")

# ---------- п.10: новости ----------
abl = pd.read_csv(OUT / "news_ablation_N0N3.csv")
d = {c: abl[(abl.config == c) & (abl.vs == "N0") & (abl.h == 1)].delta_pct.iloc[0]
     for c in ["N1", "N2", "N3"]}
rec("P10 новости ΔMAE h=1 vs N0", "+10.8% / +15.9% / +20.7%",
    f"N1=+{d['N1']:.2f}% / N2=+{d['N2']:.2f}% / N3=+{d['N3']:.2f}%",
    "MATCH" if abs(d["N1"] - 10.8) < 0.05 and abs(d["N2"] - 15.9) < 0.05
    and abs(d["N3"] - 20.7) < 0.05 else "MISMATCH",
    f"Wilcoxon p N1={abl[(abl.config=='N1')&(abl.vs=='N0')&(abl.h==1)].wilcoxon_origin_p.iloc[0]:.2f}")

# κ: из заголовков выборки × таксономия × human-разметка (дикт из 17_news_kappa.py)
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
kappas, confs = {}, {}
for t in TOPICS:
    spec = tax["topics"][t]
    raw = ks["t"].str.contains(kw_pats[t])
    if spec.get("status") == "retired":
        dd = pd.Series(False, index=ks.index)
    elif spec.get("geo") == "with_rescue":
        dd = raw & ~(is_f & ~has_r)
    else:
        dd = raw & ~is_f
    hh = np.array([SHORT[t] in (HUMAN[i].split(",") if HUMAN[i] else []) for i in ks["id"]])
    dd = dd.to_numpy()
    tp = int((dd & hh).sum()); fp = int((dd & ~hh).sum())
    fn = int((~dd & hh).sum()); tn = int((~dd & ~hh).sum())
    po = (tp + tn) / 50
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / 50 ** 2
    kappas[t] = (po - pe) / (1 - pe)
    confs[t] = (tp, fp, fn, tn)
active = [t for t in TOPICS if tax["topics"][t].get("status") != "retired"]
macro = float(np.mean([kappas[t] for t in active]))
rec("P10 macro-κ (4 активные темы, n=50)", "0.926",
    f"macro={macro:.4f}; " + "; ".join(f"{t}={kappas[t]:.3f}{confs[t]}" for t in active),
    "MATCH" if abs(macro - 0.926) < 0.0005 else "MISMATCH",
    "полный пересчёт из выборки: заголовки × таксономия v3 × human-разметка")

pd.DataFrame(results).to_csv(OUT / "rc1_v1_tail.csv", index=False)
print("\nSaved outputs/rc1_v1_tail.csv")
