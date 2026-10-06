"""Выкачка заголовков новостных архивов 2023–2024 (Коммерсантъ + Ведомости).

По D2_ru_news_sources.md: kommersant.ru/archive/news/YYYY-MM-DD (~30 топ-новостей/день,
ссылки /doc/NNNNNNN) и vedomosti.ru/archive/YYYY/MM/DD. Только заголовки+ссылки+дата —
полные тексты отдельным решением после оценки yield. Пауза 0.7с, ретраи, резюме по имеющимся файлам.
Выход: data/external/news/headlines_{kommersant,vedomosti}.jsonl
"""
import json
import re
import time
import datetime as dt
import urllib.request
from pathlib import Path

OUT = Path("/Users/said/сбериндекс/data/external/news")
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [dt.date(2023, 1, 1) + dt.timedelta(days=i) for i in range(731)]  # 2023-01-01..2024-12-31

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15"}


def fetch(url: str, tries: int = 4) -> str | None:
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except Exception:
            time.sleep(2 * (i + 1))
    return None


def parse_kommersant(html: str, day: dt.date):
    # <article class="uho rubric_lenta__item js-article" data-article-url="..." data-article-title="..." data-article-rubric-ids="...">
    out = []
    for m in re.finditer(r'<article[^>]*data-article-url="(https://www\.kommersant\.ru/doc/(\d+))"[^>]*data-article-title="([^"]+)"[^>]*>', html):
        url, doc_id, title = m.group(1), m.group(2), m.group(3)
        title = re.sub(r"\s+", " ", title).replace("&#160;", " ").replace("&quot;", '"').strip()
        rub = re.search(r'data-article-rubric-ids="([^"]*)"', m.group(0))
        if title:
            out.append({"date": day.isoformat(), "source": "kommersant",
                        "doc_id": doc_id, "title": title, "url": url,
                        "rubric_ids": rub.group(1) if rub else ""})
    seen, uniq = set(), []
    for it in out:
        if it["doc_id"] not in seen:
            seen.add(it["doc_id"])
            uniq.append(it)
    return uniq


def parse_vedomosti(html: str, day: dt.date):
    # <a href="/rubric/news/YYYY/MM/DD/ID-slug" class="article-preview-item archive-articles-list__item"...><h2 class="article-preview-item__title">TITLE</h2>
    pat = re.compile(r'<a href="(/[a-z-]+/[a-z-]+/[a-z0-9/-]*\d{4}/\d{2}/\d{2}/(\d+)[^"]*)"[^>]*class="article-preview-item[^"]*archive-articles-list__item[^"]*"[^>]*>.*?<h2[^>]*>(.*?)</h2>', re.S)
    out = []
    for m in pat.finditer(html):
        url, doc_id = m.group(1), m.group(2)
        title = re.sub(r"<[^>]+>", "", m.group(3))
        title = re.sub(r"\s+", " ", title).strip()
        if title:
            out.append({"date": day.isoformat(), "source": "vedomosti",
                        "doc_id": doc_id, "title": title,
                        "url": "https://www.vedomosti.ru" + url})
    seen, uniq = set(), []
    for it in out:
        if it["doc_id"] not in seen:
            seen.add(it["doc_id"])
            uniq.append(it)
    return uniq


def done_days(path: Path, source: str):
    if not path.exists():
        return set()
    with open(path) as f:
        return {json.loads(l)["date"] for l in f if l.strip() and json.loads(l).get("source") == source}


def crawl(source: str):
    path = OUT / f"headlines_{source}.jsonl"
    done = done_days(path, source)
    parser = parse_kommersant if source == "kommersant" else parse_vedomosti
    t0 = time.time()
    n_new = 0
    with open(path, "a") as fo:
        for i, day in enumerate(DAYS):
            if day.isoformat() in done:
                continue
            if source == "kommersant":
                url = f"https://www.kommersant.ru/archive/news/day/{day:%Y-%m-%d}"
            else:
                url = f"https://www.vedomosti.ru/archive/{day:%Y/%m/%d}"
            html = fetch(url)
            if html is None:
                print(f"[{source}] FAIL {day}", flush=True)
                continue
            items = parser(html, day)
            for it in items:
                fo.write(json.dumps(it, ensure_ascii=False) + "\n")
            fo.flush()
            n_new += 1
            if n_new % 50 == 0:
                rate = n_new / (time.time() - t0)
                print(f"[{source}] {n_new} дней, {rate:.1f} дн/с, последний {day}, накоплено строк: {sum(1 for _ in open(path))}", flush=True)
            time.sleep(0.7)
    print(f"[{source}] ГОТОВО: {n_new} новых дней", flush=True)


if __name__ == "__main__":
    crawl("kommersant")
    crawl("vedomosti")
