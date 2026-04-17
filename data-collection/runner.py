from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import pandas as pd
import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.date import DateTrigger
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager

from scraper_hkjc_live_odds import LiveScraper  
from scraper_hkjc_aspx import RacecardScraper  

ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = ROOT / "../data"
RACECARD_CSV: Path = DATA_DIR / "next-racecard.csv"

HK_TZ = pytz.timezone("Asia/Hong_Kong")
NY_TZ = pytz.timezone("America/New_York")

SCRAPE_LEAD_TIMES: List[timedelta] = [
    timedelta(hours=2),
    timedelta(hours=1),
    timedelta(minutes=45),
    timedelta(minutes=30),
] + [timedelta(minutes=m) for m in range(25, -1, -5)]  

ODDS_TYPES = ("wp", "wpq")

def log(msg: str) -> None:
    print(f"[{datetime.now(tz=NY_TZ):%Y‑%m‑%d %H:%M:%S}]  {msg}")

def _scrape_next_racecard() -> pd.DataFrame:
    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opts)

    try:
        rc = RacecardScraper(driver)
        driver.get(rc.base_url)
        df = rc.scrape_next_race_card()
    finally:
        driver.quit()

    if df.empty:
        log("⚠️ No next racecard found. Will retry in 30 minutes.")
        return pd.DataFrame()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").dt.tz_localize(HK_TZ)
    return df.dropna(subset=["datetime", "race_number"])


def fetch_next_racecard() -> pd.DataFrame:
    try:
        if (datetime.now().timestamp() - RACECARD_CSV.stat().st_mtime) < 30 * 60:
            log("📄 Using cached racecard")
            return pd.read_csv(RACECARD_CSV, parse_dates=["datetime"]).rename(columns=str.strip)
    except FileNotFoundError:
        pass

    log("🌐 Scraping fresh racecard …")
    card = _scrape_next_racecard()
    if card.empty:
        return card

    card.to_csv(RACECARD_CSV, index=False)
    log(f"🏇 Racecard saved → {RACECARD_CSV.relative_to(ROOT)}")
    return card

def run_scrape(race_no: int, odds_type: str, race_day: str) -> None:
    log(f"⏩ Scraping Race {race_no} – {odds_type} …")

    opts = webdriver.ChromeOptions()
    opts.add_argument("--headless=new")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=opts)
    scraper = LiveScraper(driver)
    driver.get(scraper.base_url)

    try:
        df = scraper.scrape_single_race(odds_type, race_no)
    finally:
        driver.quit()

    if df.empty:
        log("⚠️  Empty dataframe – nothing written")
        return

    out = DATA_DIR / f"{race_day}_{odds_type}.csv"
    write_header = not out.exists()
    df.to_csv(out, index=False, mode="a", header=write_header)
    log(f"✅ Wrote → {out.relative_to(ROOT)} ({len(df)} rows)")


def schedule_scrapes_for_card(sched: BackgroundScheduler) -> None:
    """Refresh scrape jobs for the current *next* race‑day."""
    try:
        card = fetch_next_racecard()
    except Exception as exc:
        log(f"❌ Unable to get racecard – {exc}")
        return

    if card.empty:
        log("⚠️  No upcoming racecard – skipping live‑odds scheduling.")
        return

    race_day = card["datetime"].dt.tz_convert(HK_TZ).dt.strftime("%Y‑%m‑%d").iloc[0]

    for _, row in card.drop_duplicates("race_number").iterrows():
        race_no     = int(row["race_number"])
        post_time_hk = row["datetime"]

        for lead in SCRAPE_LEAD_TIMES:
            fire_at = (post_time_hk - lead).astimezone(NY_TZ)
            if fire_at < datetime.now(tz=NY_TZ):
                continue

            for odds_type in ODDS_TYPES:
                job_id = f"{race_day}:{race_no}:{odds_type}:{int(lead.total_seconds())}"
                sched.add_job(
                    run_scrape,
                    trigger=DateTrigger(run_date=fire_at),
                    args=[race_no, odds_type, race_day],
                    id=job_id,
                    name=job_id,
                    replace_existing=True,
                )
                log(f"🔔 Scheduled {job_id} at {fire_at:%Y‑%m‑%d %H:%M:%S}")

if __name__ == "__main__":
    scheduler = BackgroundScheduler(timezone=NY_TZ)

    scheduler.add_job(fetch_next_racecard,
                      trigger="interval", minutes=30,
                      next_run_time=datetime.now(tz=NY_TZ),
                      id="update_racecard", replace_existing=True)

    scheduler.add_job(schedule_scrapes_for_card,
                      trigger="interval", minutes=15,
                      next_run_time=datetime.now(tz=NY_TZ) + timedelta(seconds=10),
                      id="schedule_scrapes", replace_existing=True,
                      args=[scheduler])

    scheduler.start()
    log("📅 Scheduler started – ctrl‑c to exit")

    try:
        while True:
            time.sleep(60)
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
        log("Shut down cleanly")