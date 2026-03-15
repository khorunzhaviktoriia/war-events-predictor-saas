import json
import re
import time
from datetime import date
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup


START_DATE = date(2022, 2, 24)
END_DATE = date(2026, 3, 1)

OUTPUT_JSON = "isw_reports.json"
ROOT = "https://understandingwar.org"
LIST_URL = ROOT + "/research/?_teams=russia-ukraine&_paged={page}"

DELAY_SEC = 3
MAX_PAGES = 300

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

TITLE_DATE_RE = re.compile(
    r"Russian Offensive Campaign Assessment,\s+([A-Za-z]+\.?)\s+(\d{1,2}),\s+(\d{4})",
    re.IGNORECASE
)

URL_DATE_RE = re.compile(
    r"russian-offensive-campaign-assessment-"
    r"(january|february|march|april|may|june|july|august|september|october|november|december)"
    r"-(\d{1,2})-(\d{4})/?$",
    re.IGNORECASE
)


def parse_title_date(title):
    if not title:
        return None

    match = TITLE_DATE_RE.search(title)
    if not match:
        return None

    month_raw, day_str, year_str = match.groups()
    month = MONTH_MAP.get(month_raw.lower().rstrip("."))
    if not month:
        return None

    return date(int(year_str), month, int(day_str))


def parse_url_date(url):
    if not url:
        return None

    match = URL_DATE_RE.search(url)
    if not match:
        return None

    month_raw, day_str, year_str = match.groups()
    month = MONTH_MAP.get(month_raw.lower())
    if not month:
        return None

    return date(int(year_str), month, int(day_str))


def fetch_soup(url, session):
    try:
        time.sleep(DELAY_SEC)
        response = session.get(url, headers=HEADERS, timeout=30)

        if response.status_code != 200:
            print(f"{response.status_code}: {url}")
            return None

        if "access denied" in response.text.lower()[:500]:
            print(f"Blocked: {url}")
            return None

        return BeautifulSoup(response.text, "html.parser")

    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def extract_text(soup):
    selectors = [
        "article .field--name-body",
        "div.field--type-text-with-summary",
        "div.node__content",
        "main article",
        "article",
        "div.field-item",
    ]

    for selector in selectors:
        block = soup.select_one(selector)
        if block:
            text = block.get_text(separator="\n", strip=True)
            if len(text) > 500:
                return text

    return ""


def collect_index(session):
    index = {}
    stale_pages = 0

    for page in range(1, MAX_PAGES + 1):
        url = LIST_URL.format(page=page)
        soup = fetch_soup(url, session)
        if soup is None:
            continue

        cards = soup.select("div.research-card-loop-item-3colgrid")
        if not cards:
            cards = soup.select(".views-row, article")

        if not cards:
            print(f"No cards on page {page}, stop.")
            break

        page_dates = []

        for card in cards:
            link = card.select_one("h2 a[href], h3 a[href], a[href]")
            if not link:
                continue

            title = link.get_text(" ", strip=True)
            if "russian offensive campaign assessment" not in title.lower():
                continue

            report_date = parse_title_date(title)
            if not report_date:
                continue

            page_dates.append(report_date)

            if START_DATE <= report_date <= END_DATE and report_date not in index:
                index[report_date] = {
                    "title": title,
                    "url": urljoin(ROOT, link.get("href", "")),
                }

        print(f"[index page {page}] collected: {len(index)}")

        if page_dates and max(page_dates) < START_DATE:
            stale_pages += 1
        else:
            stale_pages = 0

        if stale_pages >= 2:
            break

    return index


def fetch_report(report_date, meta, session):
    soup = fetch_soup(meta["url"], session)
    if soup is None:
        return None

    title_tag = soup.find("h1") or soup.find("h2")
    title = title_tag.get_text(" ", strip=True) if title_tag else meta["title"]

    real_date = parse_title_date(title)
    if real_date != report_date:
        print(f"DATE MISMATCH: expected {report_date}, got {real_date}")
        return None

    url_date = parse_url_date(meta["url"])
    if url_date is not None and url_date != report_date:
        print(f"URL DATE MISMATCH: expected {report_date}, got {url_date}")
        return None

    text = extract_text(soup)
    if not text:
        print(f"EMPTY TEXT: {meta['url']}")
        return None

    return {
        "date": report_date.isoformat(),
        "title": title,
        "url": meta["url"],
        "text": text,
    }


def validate_output(data):
    mismatches = []
    duplicate_urls = set()
    seen_urls = set()

    for row in data:
        expected = date.fromisoformat(row["date"])
        actual = parse_title_date(row["title"])

        if actual != expected:
            mismatches.append(row)

        url = row["url"]
        if url in seen_urls:
            duplicate_urls.add(url)
        seen_urls.add(url)

    print("\nValidation:")
    print(f"Records in JSON : {len(data)}")
    print(f"Duplicate URLs  : {len(duplicate_urls)}")


def main():
    session = requests.Session()

    print("Collecting links from archive pages...")
    index = collect_index(session)

    results = []
    dates = sorted(index.keys())

    for i, report_date in enumerate(dates, 1):
        record = fetch_report(report_date, index[report_date], session)
        status = "OK" if record else "BAD"
        print(f"[{i:4d}/{len(dates)}] {report_date}  {status}")

        if record:
            results.append(record)

    results.sort(key=lambda x: x["date"])

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Output: {OUTPUT_JSON}")

    validate_output(results)


if __name__ == "__main__":
    main()