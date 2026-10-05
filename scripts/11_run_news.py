"""Новостные индексы (К5): классификация заголовков по замороженной таксономии
(configs/news_taxonomy.yaml, коммит до прогона — анти-утечка), месячные/дневные агрегаты,
expanding z-score (никакой corpus-wide нормировки — R07).

Выход: outputs/news_monthly.parquet (month × topic: count, share, z_exp),
       outputs/news_daily.parquet (для burst detection × PELT стыковки),
       outputs/news_samples.md (примеры заголовков по темам — слайд обоснованности).
"""
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
    for f in NEWS.glob("headlines_*.jsonl"):
        with open(f) as fh:
            rows.extend(json.loads(l) for l in fh if l.strip())
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["title_l"] = df["title"].str.lower()
    return df


def compile_patterns(tax: dict) -> dict:
    """Ключи со '*' = префикс ('инфляц' ≈ инфляцию/инфляционный)."""
    pats = {}
    for topic, spec in tax["topics"].items():
        parts = []
        for kw in spec["keywords"]:
            kw = kw.lower()
            if kw.endswith("*"):
                parts.append(re.escape(kw[:-1]) + r"\w*")
            elif " " in kw or "*" in kw:
                parts.append(re.escape(kw).replace(r"\*", r"\w*"))
            else:
                parts.append(re.escape(kw) + r"\w*")  # слово → его словоформы
        pats[topic] = re.compile(r"(?:^|\W)(" + "|".join(parts) + r")", re.UNICODE)
    return pats


def main():
    tax = yaml.safe_load(open(ROOT / "configs" / "news_taxonomy.yaml"))
    pats = compile_patterns(tax)
    df = load_headlines()
    print(f"заголовков: {len(df)}, месяцы: {df['month'].nunique()}, источники: {df.source.unique()}")

    for topic, pat in pats.items():
        df[topic] = df["title_l"].str.contains(pat)

    # покрытие
    anyt = df[list(pats)].any(axis=1)
    print(f"покрыто хотя бы одной темой: {anyt.mean():.1%}")
    print(df[list(pats)].sum().to_string())

    # месячные агрегаты: доля темы среди новостей месяца (робастно к объёму ленты)
    months = sorted(df["month"].unique())
    agg = df.groupby("month")[list(pats)].sum()
    total = df.groupby("month").size()
    monthly = agg.div(total, axis=0)
    monthly.columns = [f"share_{c}" for c in monthly.columns]
    counts = agg.add_prefix("count_")
    out = pd.concat([counts, monthly], axis=1)
    # expanding z-score (только прошлое относительно месяца — честно для origin)
    for c in [c for c in out.columns if c.startswith("share_")]:
        vals = out[c]
        mu = vals.expanding().mean().shift(1)
        sd = vals.expanding().std().shift(1)
        out[c.replace("share_", "z_")] = (vals - mu) / sd
    out.index.name = "month"
    out.to_parquet(ROOT / "outputs" / "news_monthly.parquet")

    # дневные счётчики для burst detection
    daily = df.groupby(["date"])[list(pats)].sum()
    daily.to_parquet(ROOT / "outputs" / "news_daily.parquet")

    # примеры заголовков по темам (для отчёта/слайда)
    lines = ["# Примеры заголовков по темам (верифицировано из корпуса)\n"]
    for topic in pats:
        lines.append(f"\n## {topic}\n")
        sub = df[df[topic]].sample(min(5, df[topic].sum()), random_state=42)
        for _, r in sub.iterrows():
            lines.append(f"- ({r['date']:%Y-%m-%d}, {r['source']}) {r['title']}")
    (ROOT / "outputs" / "news_samples.md").write_text("\n".join(lines))
    print("saved outputs/news_monthly.parquet, news_daily.parquet, news_samples.md")


if __name__ == "__main__":
    main()
