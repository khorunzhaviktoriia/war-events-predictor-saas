from pathlib import Path
import subprocess
import sys
from datetime import datetime


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RECEIVER_DIR = BASE_DIR / "data_receiver"

SCRIPTS = [
    DATA_RECEIVER_DIR / "collect_absent_data.py",
    DATA_RECEIVER_DIR / "telegram_scraper_cron.py",
    DATA_RECEIVER_DIR / "get_weather_24h_OpenMeteo.py",
]


def run_script(script_path: Path) -> None:
    print("\n" + "=" * 80)
    print(f"RUNNING: {script_path.name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"{script_path.name} failed with exit code {result.returncode}")

    print(f"FINISHED: {script_path.name}")


def main():
    start_time = datetime.now()
    print(f"Collector runner started at {start_time:%Y-%m-%d %H:%M:%S}")

    for script_path in SCRIPTS:
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {script_path}")
        run_script(script_path)

    end_time = datetime.now()
    print("\n" + "=" * 80)
    print(f"All collectors finished successfully at {end_time:%Y-%m-%d %H:%M:%S}")
    print("=" * 80)


if __name__ == "__main__":
    main()