from telethon import TelegramClient
import json
import asyncio
from datetime import datetime, timezone

api_id = 12345678 #YOUR_API_ID
api_hash = "YOUR_API_HASH"

channels = [
    "DeepStateUA",
    "UkraineNow",
    "kpszsu"
]

start_date = datetime(2022, 2, 24, tzinfo=timezone.utc)

async def main():
    async with TelegramClient("telegram_session", api_id, api_hash) as client:
        all_messages = []
        counter = 0

        for channel in channels:
            async for message in client.iter_messages(channel, limit=None):

                msg_date = message.date.replace(tzinfo=timezone.utc)

                if msg_date < start_date:
                    break

                if message.text:
                    all_messages.append({
                        "date": str(message.date),
                        "channel": channel,
                        "message": message.text
                    })

                    counter += 1

                    if counter % 10000 == 0:
                        print(f"{counter} messages downloaded...")

        with open("telegram_data.json", "w", encoding="utf-8") as f:
            json.dump(all_messages, f, ensure_ascii=False, indent=4)

        print(f"Data collection finished. {len(all_messages)} messages saved to telegram_data.json")


asyncio.run(main())
