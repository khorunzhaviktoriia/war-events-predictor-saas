import json
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

def get_date():
    today = datetime.today()
    yesterday = today - timedelta(days=1)

    date = f"{yesterday.strftime('%b')} {yesterday.day}, {yesterday.year}"
    return date

def scrape_data(date):
    url = "https://understandingwar.org/research/?_teams=russia-ukraine"

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        print("RequestError:", e)

    soup = BeautifulSoup(response.text, "html.parser")
    data_list = []

    cards = soup.find_all("div", attrs={"class":"research-card-loop-item-3colgrid"})
    for card in cards:
        date_tag = card.find("p", attrs={"class":"research-card-post-date"})
        if date and date in date_tag.text:
            link = card.find("a")["href"]

            try:
                article_resp = requests.get(link, headers=headers, timeout=10)
                article_resp.raise_for_status()
            except requests.RequestException as e:
                print("RequestError:", e)

            article_soup = BeautifulSoup(article_resp.text, "html.parser")

            title_tag = article_soup.find("h1")
            title = title_tag.text.strip() if title_tag else "No title"

            data_list.append({
                "date": date,
                "title": title,
                "link": link
            })

    return data_list

def write_json(new_data):
    try:
        with open("isw_data.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    existing_links = {item["link"] for item in data}
    unique_new = [item for item in new_data if item["link"] not in existing_links]

    data.extend(unique_new)
    with open("isw_data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def main():
    yesterday = get_date()
    print(yesterday)
    data = scrape_data(yesterday)
    write_json(data)
    print(data)


if __name__=="__main__":
    main()

