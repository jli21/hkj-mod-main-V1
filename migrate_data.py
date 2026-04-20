"""
One-time migration script to reformat all existing raw data files to a clean standard.

Changes applied:
  1. Dates -> ISO YYYY-MM-DD everywhere
  2. Strip spurious index column from barrier trial CSVs
  3. Horse No. to int (not float)
  4. Drop Pla._link column
  5. Fix Running\nPosition -> Running Position
  6. Extract entity IDs from URL columns, drop full URL columns
  7. Add race_id column (YYYYMMDDRR)
  8. Flatten dividends JSON -> CSV
  9. Re-key horses JSON by horse_id, fix 2-digit years
 10. Normalize barrier trial track names

Usage:
    python migrate_data.py [--dry-run]
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "historical-data"

ASPX_DIR = DATA_DIR / "aspx-results"
BT_DIR = DATA_DIR / "barrier-trial-results"
DIV_DIR = DATA_DIR / "dividends"
HORSE_DIR = DATA_DIR / "horses"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date_ddmmyyyy(s):
    """Convert DD/MM/YYYY -> YYYY-MM-DD. Returns original if unparseable."""
    try:
        parts = s.strip().split("/")
        if len(parts) == 3:
            d, m, y = parts
            if len(y) == 2:
                y = "20" + y if int(y) < 50 else "19" + y
            return f"{y}-{m.zfill(2)}-{d.zfill(2)}"
    except Exception:
        pass
    return s


def extract_horse_id(url):
    """Extract HorseId from HKJC URL, e.g. HK_2022_H170."""
    if not isinstance(url, str):
        return None
    m = re.search(r"horseid=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_jockey_id(url):
    """Extract JockeyId from HKJC URL, e.g. WEC."""
    if not isinstance(url, str):
        return None
    m = re.search(r"jockeyid=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return m.group(1) if m else None


def extract_trainer_id(url):
    """Extract TrainerId from HKJC URL, e.g. YTP."""
    if not isinstance(url, str):
        return None
    m = re.search(r"trainerid=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return m.group(1) if m else None


def normalize_bt_track(track_str):
    """Normalize barrier trial track names to short codes."""
    if not isinstance(track_str, str):
        return track_str
    t = track_str.upper().strip()
    if "CONGHUA" in t:
        return "CG_AWT"
    if "ALL WEATHER" in t and "SHA TIN" in t:
        return "ST_AWT"
    if "ALL WEATHER" in t and "HAPPY VALLEY" in t:
        return "HV_AWT"
    if "SHA TIN" in t:
        return "ST"
    if "HAPPY VALLEY" in t:
        return "HV"
    return track_str


def make_race_id(date_iso, race_num):
    """Build race_id from ISO date and race number: YYYYMMDDRR."""
    try:
        digits = date_iso.replace("-", "")
        return f"{digits}{int(race_num):02d}"
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ASPX Results
# ---------------------------------------------------------------------------

def migrate_aspx(dry_run=False):
    print("\n=== Migrating ASPX Results ===")
    for f in sorted(ASPX_DIR.glob("aspx-results-*.csv")):
        print(f"  Processing {f.name}...")
        df = pd.read_csv(f)

        # Fix Running\nPosition header
        df.columns = [c.replace("\n", " ") for c in df.columns]

        # Date -> ISO
        if "Date" in df.columns:
            df["Date"] = df["Date"].apply(parse_date_ddmmyyyy)

        # Horse No. to int
        if "Horse No." in df.columns:
            df["Horse No."] = pd.to_numeric(df["Horse No."], errors="coerce")
            df.dropna(subset=["Horse No."], inplace=True)
            df["Horse No."] = df["Horse No."].astype(int)

        # Add race_id
        if "Date" in df.columns and "race_number" in df.columns:
            race_ids = df.apply(
                lambda r: make_race_id(r["Date"], r["race_number"]), axis=1
            )
            if "race_id" in df.columns:
                df["race_id"] = race_ids
            else:
                df.insert(0, "race_id", race_ids)

        # Extract entity IDs from links
        if "Horse_link" in df.columns:
            df["horse_id"] = df["Horse_link"].apply(extract_horse_id)
            df.drop(columns=["Horse_link"], inplace=True)

        if "Jockey_link" in df.columns:
            df["jockey_id"] = df["Jockey_link"].apply(extract_jockey_id)
            df.drop(columns=["Jockey_link"], inplace=True)

        if "Trainer_link" in df.columns:
            df["trainer_id"] = df["Trainer_link"].apply(extract_trainer_id)
            df.drop(columns=["Trainer_link"], inplace=True)

        # Drop Pla._link
        if "Pla._link" in df.columns:
            df.drop(columns=["Pla._link"], inplace=True)

        if dry_run:
            print(f"    [DRY RUN] Would write {len(df)} rows, columns: {list(df.columns)}")
        else:
            df.to_csv(f, index=False)
            print(f"    Wrote {len(df)} rows.")


# ---------------------------------------------------------------------------
# Barrier Trial Results
# ---------------------------------------------------------------------------

def migrate_barrier_trials(dry_run=False):
    print("\n=== Migrating Barrier Trial Results ===")
    for f in sorted(BT_DIR.glob("barrier-trial-results-*.csv")):
        print(f"  Processing {f.name}...")
        df = pd.read_csv(f)

        if df.empty or len(df.columns) == 0:
            print(f"    Skipping empty file.")
            continue

        # Drop spurious index column (unnamed first column)
        unnamed_cols = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

        # Fix Running\nPosition header
        df.columns = [c.replace("\n", " ") for c in df.columns]

        # Date -> ISO
        if "Date" in df.columns:
            df["Date"] = df["Date"].apply(parse_date_ddmmyyyy)

        # Normalize track names
        if "Track" in df.columns:
            df["Track"] = df["Track"].apply(normalize_bt_track)

        # Add race_id (using trial_number as the race equivalent)
        if "Date" in df.columns and "trial_number" in df.columns:
            race_ids = df.apply(
                lambda r: make_race_id(r["Date"], r["trial_number"]), axis=1
            )
            if "race_id" in df.columns:
                df["race_id"] = race_ids
            else:
                df.insert(0, "race_id", race_ids)

        # Extract horse_id from Horse_link
        if "Horse_link" in df.columns:
            df["horse_id"] = df["Horse_link"].apply(extract_horse_id)
            df.drop(columns=["Horse_link"], inplace=True)

        if dry_run:
            print(f"    [DRY RUN] Would write {len(df)} rows, columns: {list(df.columns)}")
        else:
            df.to_csv(f, index=False)
            print(f"    Wrote {len(df)} rows.")


# ---------------------------------------------------------------------------
# Dividends JSON -> CSV
# ---------------------------------------------------------------------------

def migrate_dividends(dry_run=False):
    print("\n=== Migrating Dividends (JSON -> CSV) ===")
    for f in sorted(DIV_DIR.glob("dividends-*.json")):
        print(f"  Processing {f.name}...")
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        rows = []
        for date_str, races in data.items():
            date_iso = parse_date_ddmmyyyy(date_str)
            for race_no_str, entries in races.items():
                race_id = make_race_id(date_iso, race_no_str)
                for entry in entries:
                    rows.append({
                        "race_id": race_id,
                        "date": date_iso,
                        "race_number": int(race_no_str),
                        "pool": entry.get("Pool"),
                        "combination": entry.get("Winning Combination"),
                        "dividend": entry.get("Dividend"),
                    })

        df = pd.DataFrame(rows)

        # Output as CSV with same name pattern
        out_name = f.stem + ".csv"  # e.g. dividends-2024.csv
        out_path = f.parent / out_name

        if dry_run:
            print(f"    [DRY RUN] Would write {len(df)} rows to {out_name}")
        else:
            df.to_csv(out_path, index=False)
            print(f"    Wrote {len(df)} rows to {out_name}")
            # Keep original JSON as backup
            backup = f.with_suffix(".json.bak")
            if backup.exists():
                backup.unlink()
            f.rename(backup)
            print(f"    Backed up original to {backup.name}")


# ---------------------------------------------------------------------------
# Horses JSON
# ---------------------------------------------------------------------------

def migrate_horses(dry_run=False):
    print("\n=== Migrating Horses JSON ===")
    for f in sorted(HORSE_DIR.glob("horses_HK_*.json")):
        print(f"  Processing {f.name}...")
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        new_data = {}
        for url, records in data.items():
            horse_id = extract_horse_id(url)
            if not horse_id:
                # Fallback: use the URL as key
                horse_id = url

            for record in records:
                # Fix 2-digit year dates
                if "Date" in record and isinstance(record["Date"], str):
                    record["Date"] = parse_date_ddmmyyyy(record["Date"])

                # Extract IDs from link columns in records
                if "Trainer_link" in record:
                    record["trainer_id"] = extract_trainer_id(record["Trainer_link"])
                    del record["Trainer_link"]

                if "Jockey_link" in record:
                    record["jockey_id"] = extract_jockey_id(record["Jockey_link"])
                    del record["Jockey_link"]

                if "Race Index_link" in record:
                    del record["Race Index_link"]

            new_data[horse_id] = records

        if dry_run:
            print(f"    [DRY RUN] Would write {len(new_data)} horses")
        else:
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(new_data, fh, indent=2)
            print(f"    Wrote {len(new_data)} horses.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Migrate raw data to clean format")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without writing files")
    args = parser.parse_args()

    print(f"Data directory: {DATA_DIR}")
    print(f"Dry run: {args.dry_run}")

    migrate_aspx(dry_run=args.dry_run)
    migrate_barrier_trials(dry_run=args.dry_run)
    migrate_dividends(dry_run=args.dry_run)
    migrate_horses(dry_run=args.dry_run)

    print("\n=== Migration complete ===")


if __name__ == "__main__":
    main()
