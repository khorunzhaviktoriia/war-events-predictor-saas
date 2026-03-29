from telethon import TelegramClient
import json
import asyncio
import os
from datetime import datetime, timezone

api_id = 12345678  # YOUR_API_ID
api_hash = "YOUR_API_HASH"

channels = [
    "DeepStateUA",
    "UkraineNow",
    "kpszsu"
]

json_file = "telegram_data.json"
default_start_date = datetime(2022, 2, 24, tzinfo=timezone.utc)


def load_existing_data():
    if os.path.exists(json_file):
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def parse_message_date(date_str):
    return datetime.fromisoformat(date_str)


def get_last_dates_by_channel(existing_data):
    last_dates = {}

    for item in existing_data:
        channel = item.get("channel")
        date_str = item.get("date")

        if not channel or not date_str:
            continue

        msg_date = parse_message_date(date_str)

        if channel not in last_dates or msg_date > last_dates[channel]:
            last_dates[channel] = msg_date

    return last_dates


async def main():
    existing_data = load_existing_data()
    last_dates = get_last_dates_by_channel(existing_data)

    new_messages = []
    counter = 0

    async with TelegramClient("telegram_session", api_id, api_hash) as client:
        for channel in channels:
            channel_start_date = last_dates.get(channel, default_start_date)

            print(f"\nChannel: {channel}")
            print(f"Loading messages after: {channel_start_date}")

            # reverse=True -> від старих до нових
            async for message in client.iter_messages(channel, limit=None, reverse=True):
                if not message.text:
                    continue

                msg_date = message.date
                if msg_date.tzinfo is None:
                    msg_date = msg_date.replace(tzinfo=timezone.utc)

                if msg_date <= channel_start_date:
                    continue

                new_messages.append({
                    "date": msg_date.isoformat(),
                    "channel": channel,
                    "message": message.text
                })

                counter += 1

                if counter % 1000 == 0:
                    print(f"{counter} new messages downloaded...")

    all_messages = existing_data + new_messages

    seen = set()
    unique_messages = []

    for msg in all_messages:
        key = (msg["channel"], msg["date"], msg["message"])
        if key not in seen:
            seen.add(key)
            unique_messages.append(msg)

    unique_messages.sort(key=lambda x: (x["channel"], x["date"]))

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(unique_messages, f, ensure_ascii=False, indent=4)

    print(f"\nFinished.")
    print(f"Added {len(new_messages)} new messages.")
    print(f"Total saved: {len(unique_messages)}")
    print(f"Data saved to {json_file}")


asyncio.run(main())