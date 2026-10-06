"""Новостные индексы (К5): классификация заголовков по таксономии v3
(configs/news_taxonomy.yaml — версии в шапке файла), месячные/дневные агрегаты,
expanding z-score (никакой corpus-wide нормировки — R07).

Изменения v3 (06.10.2026, по аудиту rc0 — A12_news_auditor / A03_kuznetsov):
- geo-фильтр: иностранные топонимы снимают хит с темы (политика exclude_only /
  with_rescue по теме) — до этого ~15% хитов были вне канала РФ-потребления;
- темы со status: retired (fx_ruble, consumer_sentiment) исключены из фичей —
  вырожденный объём (37 и 3 хита за 24 мес при topic_min_total=48);
- z-score считается только при count >= min_volume.z_min_count (5), иначе NaN —
  z-пики на 2-7 заголовках это пуассоновский шум; inf при sd=0 -> NaN;
- диагностика мультилейбла и гео-очистки печатается в лог прогона.

Выход: outputs/news_monthly_v2.parquet (month × topic: count, share, z_exp),
       outputs/news_daily_v2.parquet,
       outputs/news_samples_v2.md (примеры заголовков по активным темам).
"""
import html
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
NEWS = ROOT / "data" / "external" / "news"


def load_headlines() -> pd.DataFrame:
    rows = []
    files = list(NEWS.glob("headlines_*.jsonl"))
    if not files:
        raise SystemExit(
            f"Новостный корпус не найден в {NEWS}. Сначала: make news-corpus "
            f"(краулер scripts/crawl_news_headlines.py, ~25–40 мин).")
    for f in files:
        with open(f) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["title"] = df["title"].map(html.unescape)  # &#171; -> « (косметика корпуса, аудит P2)
    df["title_l"] = df["title"].str.lower()
    return df


def _stem_regex(kw: str) -> str:
    """Стем -> regex: '*' = префикс ('инфляц' ~ инфляцию/инфляционный),
    пробелы = фраза, последний токен фразы тоже префиксный."""
    kw = kw.lower()
    parts = []
    for tok in kw.split():
        if tok.endswith("*"):
            parts.append(re.escape(tok[:-1]) + r"\w*")
        else:
            parts.append(re.escape(tok) + r"\w*")
    return r"\s+".join(parts)


def compile_patterns(tax: dict) -> dict:
    pats = {}
    for topic, spec in tax["topics"].items():
        parts = [_stem_regex(kw) for kw in spec["keywords"]]
        pats[topic] = re.compile(r"(?:^|\W)(" + "|".join(parts) + r")", re.UNICODE)
    return pats


def compile_geo(tax: dict):
    excl = re.compile(r"(?:^|\W)(" + "|".join(_stem_regex(s) for s in tax["geo_exclude"]) + r")", re.UNICODE)
    resc = re.compile(r"(?:^|\W)(" + "|".join(_stem_regex(s) for s in tax["geo_rescue"]) + r")", re.UNICODE)
    return excl, resc


def main():
    tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
    pats = compile_patterns(tax)
    geo_excl, geo_resc = compile_geo(tax)
    mv = tax["min_volume"]
    df = load_headlines()
    print(f"заголовков: {len(df)}, месяцы: {df['month'].nunique()}, источники: {df.source.unique()}")

    is_foreign = df["title_l"].str.contains(geo_excl)
    has_rescue = df["title_l"].str.contains(geo_resc)
    print(f"geo-фильтр: заголовков с иностранным топонимом: {is_foreign.sum()} "
          f"({is_foreign.mean():.1%} корпуса), из них с RU-маркером (rescue): {(is_foreign & has_rescue).sum()}")

    active, stat_rows = [], []
    for topic, spec in tax["topics"].items():
        raw = df["title_l"].str.contains(pats[topic])
        if spec.get("geo") == "with_rescue":
            clean = raw & ~(is_foreign & ~has_rescue)
        else:  # exclude_only
            clean = raw & ~is_foreign
        retired = spec.get("status") == "retired"
        stat_rows.append(dict(topic=topic, status=spec.get("status", "active"),
                              hits_raw=int(raw.sum()), hits_geo=int(clean.sum()),
                              geo_removed=int((raw & ~clean).sum()),
                              months_nz=int((clean.groupby(df["month"]).sum() > 0).sum())))
        if not retired:
            df[topic] = clean
            active.append(topic)
        else:
            df[topic] = False  # retired: не входит в фичи, диагностика выше
    stats = pd.DataFrame(stat_rows).set_index("topic")
    print("\n=== хиты по темам: до geo / после geo / снято / ненулевых месяцев ===")
    print(stats.to_string())

    # покрытие активными темами
    anyt = df[active].any(axis=1)
    print(f"\nпокрыто активными темами: {anyt.mean():.1%} ({int(anyt.sum())} заголовков)")
    n_multi = (df[active].sum(axis=1) > 1).sum()
    print(f"мультилейбл (активные темы): {n_multi} заголовков ({n_multi / len(df):.3%}) — "
          f"лексиконы тем почти дизъюнктны, это свойство дизайна, не баг")
    # канонический мультилейбл-кейс из R07 должен ловиться >= 2 темами
    canon = df["title_l"].str.contains("ключевая ставка – 12%")
    if canon.any():
        hit_topics = [t for t in active if (df.loc[canon, t]).any()]
        print(f"канонический кейс R07 «Ключевая ставка – 12%…»: темы = {hit_topics}")

    # месячные агрегаты: доля темы среди новостей месяца (робастно к объёму ленты)
    agg = df.groupby("month")[active].sum()
    total = df.groupby("month").size()
    monthly = agg.div(total, axis=0)
    monthly.columns = [f"share_{c}" for c in monthly.columns]
    counts = agg.add_prefix("count_")
    out = pd.concat([counts, monthly], axis=1)
    # expanding z-score (только прошлое относительно месяца — честно для origin);
    # z = NaN при count < z_min_count (пуассоновский шум) и при sd=0 (раньше давало inf)
    for c in [c for c in out.columns if c.startswith("share_")]:
        topic = c[len("share_"):]
        vals = out[c]
        mu = vals.expanding().mean().shift(1)
        sd = vals.expanding().std().shift(1)
        z = (vals - mu) / sd.replace(0, np.nan)
        z = z.mask(out[f"count_{topic}"] < mv["z_min_count"])
        out[c.replace("share_", "z_")] = z
    out.index.name = "month"
    out.to_parquet(ROOT / "outputs" / "news_monthly_v2.parquet")

    # дневные счётчики для burst detection
    daily = df.groupby(["date"])[active].sum()
    daily.to_parquet(ROOT / "outputs" / "news_daily_v2.parquet")

    # примеры заголовков по активным темам (для отчёта/слайда)
    lines = ["# Примеры заголовков по темам (v3, после geo-фильтра; верифицировано из корпуса)\n"]
    for topic in active:
        lines.append(f"\n## {topic}\n")
        sub = df[df[topic]].sample(min(5, int(df[topic].sum())), random_state=42)
        for _, r in sub.iterrows():
            lines.append(f"- ({r['date']:%Y-%m-%d}, {r['source']}) {r['title']}")
    (ROOT / "outputs" / "news_samples_v2.md").write_text("\n".join(lines))
    print("saved outputs/news_monthly_v2.parquet, news_daily_v2.parquet, news_samples_v2.md")


if __name__ == "__main__":
    main()
