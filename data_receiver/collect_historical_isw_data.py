import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import time
import os
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_HERE, "isw_cache")
OUTPUT_JSON = os.path.join(_HERE, "isw_reports.json")

START_DATE = date(2022, 2, 24)
END_DATE = date(2022, 3, 24)
DELAY_SEC = 5.0

MONTHS = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://understandingwar.org/",
}

ROOT = "https://understandingwar.org"

def build_urls(d: date) -> list[str]:
    m, day, year = MONTHS[d.month], d.day, d.year
    urls = [
        f"{ROOT}/research/russia-ukraine/russian-offensive-campaign-assessment-{m}-{day}-{year}",
        f"{ROOT}/backgrounder/russian-offensive-campaign-assessment-{m}-{day}",
        f"{ROOT}/backgrounder/russian-offensive-campaign-assessment-{m}-{day}-{year}",
    ]
    if year >= 2025:
        urls.insert(0, f"{ROOT}/research/russia-ukraine/russian-offensive-campaign-assessment-{m}-{day}-{year}/")
    return urls

def extract_text(soup: BeautifulSoup) -> str:
    for sel in [
        "div.field--type-text-with-summary",
        "div.field-item",
        "article .field--name-body",
        "div.node__content",
        "main article",
        "article",
    ]:
        block = soup.select_one(sel)
        if block:
            text = block.get_text(separator="\n", strip=True)
            if len(text) > 300:
                return text
    return ""

def is_real_report(soup: BeautifulSoup, title: str) -> bool:
    if not title:
        return False
    page_title = (soup.title.string or "") if soup.title else ""
    if "page not found" in page_title.lower():
        return False
    if not any(k in title.lower() for k in ("assessment", "offensive", "campaign")):
        return False
    return True

def fetch_report(d: date, session: requests.Session) -> dict | None:
    for url in build_urls(d):
        try:
            time.sleep(DELAY_SEC)
            resp = session.get(url, headers=HEADERS, timeout=25, allow_redirects=True)

            if resp.status_code == 403 or "access denied" in resp.text.lower()[:500]:
                print("\nIP banned")
                raise SystemExit(1)

            if resp.status_code != 200:
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            title_tag = soup.find("h1") or soup.find("h2")
            title = title_tag.get_text(strip=True) if title_tag else ""

            if not is_real_report(soup, title):
                continue

            text = extract_text(soup)
            if not text:
                continue

            return {
                "date":  d.isoformat(),
                "title": title,
                "url":   resp.url,
                "text":  text,
            }

        except SystemExit:
            raise
        except requests.exceptions.ReadTimeout:
            pass
        except Exception as e:
            print(f" {d} error: {e}")

    return None


def date_range(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def load_hits() -> set:
    hits = set()
    if not os.path.isdir(CACHE_DIR):
        return hits
    for fname in os.listdir(CACHE_DIR):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CACHE_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            if data:
                hits.add(fname[:-5])
        except Exception:
            pass
    return hits


def assemble_output(all_dates: list) -> int:
    results = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if not fname.endswith(".json"):
            continue
        try:
            with open(os.path.join(CACHE_DIR, fname), encoding="utf-8") as f:
                data = json.load(f)
            if data:
                results.append(data)
        except Exception:
            pass
    results.sort(key=lambda r: r["date"])
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return len(results)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    session = requests.Session()
    all_dates = list(date_range(START_DATE, END_DATE))
    hits = load_hits()
    todo = [d for d in all_dates if d.isoformat() not in hits]
    new_hits = 0

    for i, d in enumerate(todo, 1):
        date_str   = d.isoformat()
        cache_path = os.path.join(CACHE_DIR, f"{date_str}.json")

        report = fetch_report(d, session)
        status = "OK" if report else "-"
        print(f"  [{i:4d}/{len(todo)}] {date_str}  {status}")

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        if report:
            new_hits += 1

    total = assemble_output(all_dates)
    size_mb = os.path.getsize(OUTPUT_JSON) / 1024 / 1024

    print(f"\nDone.")
    print(f"  Total in JSON: {total} / {len(all_dates)}")
    print(f"  Output       : {OUTPUT_JSON}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()