from telethon import TelegramClient
import json
import asyncio
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import os

api_id = int(os.getenv("TG_API_ID"))
api_hash = os.getenv("TG_API_HASH")

channels = [
    "DeepStateUA",
    "UkraineNow",
    "kpszsu"
]

BASE_DIR = Path(__file__).resolve().parent.parent
PARQUET_PATH = BASE_DIR / "data" / "final_merged_dataset.parquet"
SNAPSHOTS_DIR = BASE_DIR / "data" / "raw_snapshots" / "telegram"
SESSION_NAME = "telegram_session"
PARQUET_DATE_COLUMN = "datetime_hour"
KYIV_TZ = ZoneInfo("Europe/Kyiv")

def get_snapshot_path(dt: datetime) -> Path:
    date_str = dt.strftime("%Y-%m-%d")
    hour_str = dt.strftime("%Y-%m-%d_%H-00")
    return SNAPSHOTS_DIR / date_str / f"telegram_{hour_str}.json"


def load_snapshot(path: Path) -> list:
    if path.exists():
        with open(path,"r",encoding="utf-8") as f:
            return json.load(f)
    return []


def save_snapshot(path: Path, messages: list):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def get_last_date() -> datetime | None:
    if not PARQUET_PATH.exists():
        print(f"final_merged_dataset.parquet not found: {PARQUET_PATH}")
        return None

    df = pd.read_parquet(PARQUET_PATH, columns=[PARQUET_DATE_COLUMN])
    last_date = df[PARQUET_DATE_COLUMN].max()

    if last_date.tzinfo is None:
        last_date = last_date.tz_localize(KYIV_TZ)
    else:
        last_date = last_date.tz_convert(KYIV_TZ)

    return last_date.to_pydatetime()


def get_missing_hours(last_date: datetime, now: datetime) -> list[datetime]:
    start = last_date.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    current_hour = now.replace(minute=0, second=0, microsecond=0)

    hours = []
    iter_hour = start
    while iter_hour < current_hour:
        hours.append(iter_hour)
        iter_hour += timedelta(hours=1)

    return hours


async def collect_messages_for_period(client: TelegramClient, date_from: datetime, date_to: datetime) -> list[dict]:
    messages = []

    for channel in channels:
        print(f"  {channel}: collecting from {date_from} to {date_to}")

        async for message in client.iter_messages(channel, limit=None,offset_date=date_to):
            if not message.text:
                continue

            msg_date = message.date
            if msg_date.tzinfo is None:
                msg_date = msg_date.replace(tzinfo=timezone.utc)
            msg_date = msg_date.astimezone(KYIV_TZ)

            if msg_date < date_from:
                break

            messages.append({
                "date": msg_date.isoformat(),
                "channel": channel,
                "message": message.text
            })

    return messages


async def run_backfill(client: TelegramClient, missing_hours: list[datetime]):
    total = len(missing_hours)
    print(f"\nNeed to fill {total} hours...")

    for i, hour_start in enumerate(missing_hours):
        hour_end = hour_start + timedelta(hours=1)
        snapshot_path = get_snapshot_path(hour_start)

        if snapshot_path.exists():
            print(f"  [{i+1}/{total}] Already exists, skipping: {snapshot_path}")
            continue

        print(f"  [{i+1}/{total}] Collecting: {hour_start.strftime('%Y-%m-%d %H:00')}...")

        messages = await collect_messages_for_period(client, hour_start, hour_end)

        save_snapshot(snapshot_path, messages)
        print(f"    Saved {len(messages)} messages to {snapshot_path}")


async def run_current_hour(client: TelegramClient, now: datetime):
    hour_start = now.replace(minute=0, second=0, microsecond=0)
    hour_end = now

    snapshot_path = get_snapshot_path(hour_start)

    print(f"\n Collecting from {hour_start.strftime('%Y-%m-%d %H:00')}...")

    existing = load_snapshot(snapshot_path)
    existing_keys = {(m["channel"], m["date"]) for m in existing}

    new_messages = await collect_messages_for_period(client, hour_start, hour_end)

    unique_new = [
        m for m in new_messages
        if (m["channel"], m["date"]) not in existing_keys
    ]

    all_messages = existing + unique_new
    all_messages.sort(key=lambda x: (x["channel"], x["date"]))

    save_snapshot(snapshot_path, all_messages)
    print(f"  Saved {len(all_messages)} messages ({len(unique_new)} new)")


async def main():
    now = datetime.now(KYIV_TZ)
    last_date = get_last_date()
    print(f"START: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if last_date is None:
        print("final_merged_dataset.parquet is missing. No backfill source.")
        missing_hours = []
    else:
        print(f"Last date in final_merged_dataset.parquet: {last_date.strftime('%Y-%m-%d %H:%M')}")
        missing_hours = get_missing_hours(last_date, now)

    async with TelegramClient(SESSION_NAME, api_id, api_hash) as client:
        if missing_hours:
            await run_backfill(client, missing_hours)
        else:
            print("No missing full hours")

    print("\nScript finished running.")

asyncio.run(main())
