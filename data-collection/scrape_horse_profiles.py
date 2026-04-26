from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scrape_hkjc_common import (
    DATA_DIR,
    absolute_url,
    collect_horse_ids,
    ensure_dir,
    fetch_html,
    log,
    normalize_space,
    parse_ddmmyyyy,
    run_parallel,
    scrape_timestamp,
)


OUTPUT_DIR = DATA_DIR / "horse-profiles"
OUTPUT_PATH = OUTPUT_DIR / "horse-profiles.csv"
BASE_URL = "https://racing.hkjc.com/en-us/local/information/horse"
PROFILE_LABELS = {
    "Country of Origin / Age",
    "Country of Origin",
    "Colour / Sex",
    "Import Type",
    "Season Stakes",
    "Total Stakes",
    "No. of 1-2-3-Starts",
    "No. of starts in past 10 race meetings",
    "Current Stable Location (Arrival Date)",
    "Import Date",
    "Trainer",
    "Owner",
    "Current Rating",
    "Start of Season Rating",
    "Last Rating",
    "Sire",
    "Dam",
    "Dam's Sire",
    "Same Sire",
}


def horse_url(horse_id: str) -> str:
    return f"{BASE_URL}?horseid={horse_id}"


def clean_label(value: str) -> str:
    return normalize_space(value).replace("*", "")


def parse_header(soup: BeautifulSoup) -> tuple[str | None, str | None, str]:
    for table in soup.find_all("table"):
        for tr in table.find_all("tr")[:4]:
            for cell in tr.find_all(["td", "th"]):
                text = normalize_space(cell.get_text(" ", strip=True))
                match = re.fullmatch(r"(.+?) \(([A-Z0-9]+)\)(?: \((Retired)\))?", text)
                if match:
                    horse_name = match.group(1)
                    brand_no = match.group(2)
                    status = "retired" if match.group(3) else "active"
                    return horse_name, brand_no, status
    title = normalize_space(soup.title.get_text(" ", strip=True) if soup.title else "")
    title_match = re.match(r"(.+?) - Horses", title)
    horse_name = title_match.group(1) if title_match else None
    return horse_name, None, "unknown"


def extract_profile_map(soup: BeautifulSoup) -> dict[str, str]:
    best: dict[str, str] = {}
    best_hits = -1
    for table in soup.find_all("table"):
        current: dict[str, str] = {}
        hits = 0
        for tr in table.find_all("tr"):
            cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
            if len(cells) >= 3 and cells[1] == ":":
                label = clean_label(cells[0])
                if label in PROFILE_LABELS:
                    current[label] = cells[2]
                    hits += 1
        if hits > best_hits:
            best = current
            best_hits = hits
    return best


def split_slash_pair(value: str | None) -> tuple[str | None, str | None]:
    text = normalize_space(value)
    if not text:
        return None, None
    parts = [normalize_space(part) for part in text.split("/")]
    if len(parts) >= 2:
        return parts[0] or None, parts[1] or None
    return text, None


def split_location_date(value: str | None) -> tuple[str | None, str | None]:
    text = normalize_space(value)
    if not text:
        return None, None
    match = re.match(r"(.+?) \((\d{2}/\d{2}/\d{4})\)", text)
    if match:
        return normalize_space(match.group(1)), parse_ddmmyyyy(match.group(2))
    return text, None


def split_record_counts(value: str | None) -> tuple[str | None, str | None, str | None, str | None]:
    text = normalize_space(value)
    if not text:
        return None, None, None, None
    parts = [normalize_space(part) for part in text.split("-")]
    if len(parts) == 4:
        return tuple(part or None for part in parts)  # type: ignore[return-value]
    return text, None, None, None


def scrape_one(horse_id: str) -> dict[str, object]:
    url = horse_url(horse_id)
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    profile = extract_profile_map(soup)
    horse_name, brand_no, status = parse_header(soup)

    country_of_origin, age = split_slash_pair(profile.get("Country of Origin / Age") or profile.get("Country of Origin"))
    colour, sex = split_slash_pair(profile.get("Colour / Sex"))
    stable_location, stable_arrival_date = split_location_date(profile.get("Current Stable Location (Arrival Date)"))
    wins, seconds, thirds, starts = split_record_counts(profile.get("No. of 1-2-3-Starts"))

    return {
        "horse_id": horse_id,
        "horse_name": horse_name,
        "brand_no": brand_no,
        "status": status,
        "country_of_origin": country_of_origin,
        "age": age,
        "colour": colour,
        "sex": sex,
        "import_type": profile.get("Import Type") or None,
        "import_date": parse_ddmmyyyy(profile.get("Import Date")),
        "current_trainer": profile.get("Trainer") or None,
        "owner": profile.get("Owner") or None,
        "season_stakes": profile.get("Season Stakes") or None,
        "total_stakes": profile.get("Total Stakes") or None,
        "record_1_2_3_starts": profile.get("No. of 1-2-3-Starts") or None,
        "wins": wins,
        "seconds": seconds,
        "thirds": thirds,
        "starts": starts,
        "starts_past_10_meetings": profile.get("No. of starts in past 10 race meetings") or None,
        "stable_location": stable_location,
        "stable_arrival_date": stable_arrival_date,
        "current_rating": profile.get("Current Rating") or None,
        "start_season_rating": profile.get("Start of Season Rating") or None,
        "last_rating": profile.get("Last Rating") or None,
        "sire": profile.get("Sire") or None,
        "dam": profile.get("Dam") or None,
        "damsire": profile.get("Dam's Sire") or None,
        "same_sire": profile.get("Same Sire") or None,
        "source_url": url,
        "scrape_ts": scrape_timestamp(),
    }


def existing_horse_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["horse_id"], dtype=str)
    except Exception:
        return set()
    return set(df["horse_id"].dropna().astype(str).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape HKJC horse profile metadata for all known horse IDs.")
    parser.add_argument("--years", nargs="+", type=int, help="Restrict horse universe to result years")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=250)
    args = parser.parse_args()

    horse_ids = collect_horse_ids(args.years)
    if not horse_ids:
        raise SystemExit("No horse IDs found from historical results.")

    if args.output.exists() and not args.overwrite and not args.resume:
        raise SystemExit(f"Output already exists: {args.output}")

    ensure_dir(args.output.parent)
    failure_path = args.output.with_suffix(".failed.txt")
    write_header = args.overwrite or not args.output.exists()
    if args.overwrite and args.output.exists():
        args.output.unlink()
    if args.overwrite and failure_path.exists():
        failure_path.unlink()

    completed = existing_horse_ids(args.output) if args.resume else set()
    pending = [horse_id for horse_id in horse_ids if horse_id not in completed]
    log(f"Horse profile universe={len(horse_ids):,}, completed={len(completed):,}, pending={len(pending):,}")

    total_written = len(completed)
    all_failures: list[tuple[object, str]] = []
    batch_size = max(1, args.batch_size)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results, failures = run_parallel(batch, scrape_one, args.workers, "horse-profiles", progress_every=100)
        all_failures.extend(failures)
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values(["horse_id"]).reset_index(drop=True)
            df.to_csv(args.output, index=False, mode="a", header=write_header)
            write_header = False
            total_written += len(df)
            log(f"Appended {len(df):,} profile rows; total written={total_written:,}")
    if all_failures:
        with open(failure_path, "a", encoding="utf-8") as fh:
            for item, reason in all_failures:
                fh.write(f"{item}\t{reason}\n")
    log(f"Wrote {total_written:,} horse profile rows -> {args.output}")
    if all_failures:
        log(f"Recorded {len(all_failures)} failures -> {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
