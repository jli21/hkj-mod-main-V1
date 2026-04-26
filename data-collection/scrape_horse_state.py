from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scrape_hkjc_common import (
    DATA_DIR,
    collect_horse_ids,
    ensure_dir,
    fetch_html,
    log,
    normalize_space,
    parse_ddmmyyyy,
    run_parallel,
    scrape_timestamp,
)


OUTPUT_DIR = DATA_DIR
MOVE_OUTPUT = OUTPUT_DIR / "horse-movement-records" / "horse-movement-records.csv"
TRACKWORK_DIR = OUTPUT_DIR / "horse-trackwork-records"
VET_OUTPUT = OUTPUT_DIR / "horse-veterinary-records" / "horse-veterinary-records.csv"
PAGE_CHOICES = ["movement", "trackwork", "vet"]

MOVEMENT_URL = "https://racing.hkjc.com/en-us/local/information/movementrecords?horseid={horse_id}"
TRACKWORK_URL = "https://racing.hkjc.com/en-us/local/information/trackworkresult?horseid={horse_id}"
VET_URL = "https://racing.hkjc.com/en-us/local/information/ovehorse?horseid={horse_id}"


def table_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    return rows


def scrape_movement(horse_id: str) -> list[dict[str, object]]:
    url = MOVEMENT_URL.format(horse_id=horse_id)
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    scrape_ts = scrape_timestamp()
    for table in soup.find_all("table"):
        rows = table_rows(table)
        if rows and rows[0][:3] == ["From", "To", "Arrival Date"]:
            out: list[dict[str, object]] = []
            for row in rows[1:]:
                padded = row + [""] * (3 - len(row))
                out.append(
                    {
                        "horse_id": horse_id,
                        "from_location": padded[0] or None,
                        "to_location": padded[1] or None,
                        "arrival_date": parse_ddmmyyyy(padded[2]),
                        "source_url": url,
                        "scrape_ts": scrape_ts,
                    }
                )
            return out
    return []


def scrape_trackwork(horse_id: str) -> list[dict[str, object]]:
    url = TRACKWORK_URL.format(horse_id=horse_id)
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    scrape_ts = scrape_timestamp()
    for table in soup.find_all("table"):
        rows = table_rows(table)
        if rows and rows[0][:5] == ["Date", "Type", "Racecourse/Track", "Workouts", "Gear"]:
            out: list[dict[str, object]] = []
            for row in rows[1:]:
                padded = row + [""] * (5 - len(row))
                out.append(
                    {
                        "horse_id": horse_id,
                        "work_date": parse_ddmmyyyy(padded[0]),
                        "work_type": padded[1] or None,
                        "racecourse_track": padded[2] or None,
                        "workouts": padded[3] or None,
                        "gear": padded[4] or None,
                        "source_url": url,
                        "scrape_ts": scrape_ts,
                    }
                )
            return out
    return []


def scrape_vet(horse_id: str) -> list[dict[str, object]]:
    url = VET_URL.format(horse_id=horse_id)
    soup = BeautifulSoup(fetch_html(url), "html.parser")
    scrape_ts = scrape_timestamp()
    for table in soup.find_all("table"):
        rows = table_rows(table)
        if rows and rows[0][:3] == ["Date", "Details", "Passed Date"]:
            out: list[dict[str, object]] = []
            for row in rows[1:]:
                padded = row + [""] * (3 - len(row))
                out.append(
                    {
                        "horse_id": horse_id,
                        "record_date": parse_ddmmyyyy(padded[0]),
                        "details": padded[1] or None,
                        "passed_date": parse_ddmmyyyy(padded[2]),
                        "source_url": url,
                        "scrape_ts": scrape_ts,
                    }
                )
            return out
    return []


def scrape_one(horse_id: str) -> dict[str, list[dict[str, object]]]:
    return {
        "movement": scrape_movement(horse_id),
        "trackwork": scrape_trackwork(horse_id),
        "vet": scrape_vet(horse_id),
    }


def make_worker(pages: list[str]):
    selected = list(pages)

    def worker(horse_id: str) -> dict[str, list[dict[str, object]]]:
        payload = {"movement": [], "trackwork": [], "vet": []}
        if "movement" in selected:
            payload["movement"] = scrape_movement(horse_id)
        if "trackwork" in selected:
            payload["trackwork"] = scrape_trackwork(horse_id)
        if "vet" in selected:
            payload["vet"] = scrape_vet(horse_id)
        return payload

    return worker


def flatten(results: list[dict[str, list[dict[str, object]]]], key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.extend(result[key])
    return pd.DataFrame(rows)


def trackwork_output(year: str) -> Path:
    return TRACKWORK_DIR / f"horse-trackwork-records-{year}.csv"


def existing_trackwork_ids() -> set[str]:
    horse_ids: set[str] = set()
    for path in TRACKWORK_DIR.glob("horse-trackwork-records-*.csv"):
        horse_ids.update(existing_horse_ids(path))
    return horse_ids


def trackwork_existing_rows() -> int:
    total = 0
    for path in TRACKWORK_DIR.glob("horse-trackwork-records-*.csv"):
        total += max(0, sum(1 for _ in open(path, encoding="utf-8")) - 1)
    return total


def existing_horse_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["horse_id"], dtype=str)
    except Exception:
        return set()
    return set(df["horse_id"].dropna().astype(str).tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape HKJC horse movement, trackwork, and veterinary pages.")
    parser.add_argument("--years", nargs="+", type=int, help="Restrict horse universe to result years")
    parser.add_argument("--pages", nargs="+", choices=PAGE_CHOICES, default=PAGE_CHOICES)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    horse_ids = collect_horse_ids(args.years)
    if not horse_ids:
        raise SystemExit("No horse IDs found from historical results.")

    output_map = {
        "movement": MOVE_OUTPUT,
        "trackwork": TRACKWORK_DIR,
        "vet": VET_OUTPUT,
    }
    selected_pages = list(dict.fromkeys(args.pages))
    selected_outputs = [output_map[page] for page in selected_pages]

    for path in selected_outputs:
        if path.exists() and not args.overwrite and not args.resume:
            raise SystemExit(f"Output already exists: {path}")
        ensure_dir(path if path == TRACKWORK_DIR else path.parent)

    failure_path = VET_OUTPUT.with_suffix(".failed.txt")
    if args.overwrite:
        for path in selected_outputs + [failure_path]:
            if path == TRACKWORK_DIR and path.exists():
                for csv_path in path.glob("horse-trackwork-records-*.csv"):
                    csv_path.unlink()
                legacy_combined = path / "horse-trackwork-records.csv"
                if legacy_combined.exists():
                    legacy_combined.unlink()
            elif path.exists():
                path.unlink()

    resume_anchor = selected_outputs[0] if selected_outputs else MOVE_OUTPUT
    if args.resume and resume_anchor == TRACKWORK_DIR:
        completed = existing_trackwork_ids()
    else:
        completed = existing_horse_ids(resume_anchor) if args.resume else set()
    pending = [horse_id for horse_id in horse_ids if horse_id not in completed]
    log(
        f"Horse state pages={','.join(selected_pages)} universe={len(horse_ids):,}, "
        f"completed={len(completed):,}, pending={len(pending):,}"
    )

    move_header = not MOVE_OUTPUT.exists()
    vet_header = not VET_OUTPUT.exists()
    total_move = 0 if "movement" not in selected_pages or move_header else sum(1 for _ in open(MOVE_OUTPUT, encoding="utf-8")) - 1
    total_track = 0 if "trackwork" not in selected_pages else trackwork_existing_rows()
    total_vet = 0 if "vet" not in selected_pages or vet_header else sum(1 for _ in open(VET_OUTPUT, encoding="utf-8")) - 1
    track_headers: dict[str, bool] = {}
    all_failures: list[tuple[object, str]] = []
    batch_size = max(1, args.batch_size)
    worker = make_worker(selected_pages)
    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        results, failures = run_parallel(batch, worker, args.workers, "horse-state", progress_every=50)
        all_failures.extend(failures)

        move_df = flatten(results, "movement")
        if "movement" in selected_pages and not move_df.empty:
            move_df = move_df.sort_values(["horse_id", "arrival_date", "from_location", "to_location"]).reset_index(drop=True)
            move_df.to_csv(MOVE_OUTPUT, index=False, mode="a", header=move_header)
            move_header = False
            total_move += len(move_df)

        track_df = flatten(results, "trackwork")
        if "trackwork" in selected_pages and not track_df.empty:
            track_df = track_df.sort_values(["horse_id", "work_date", "work_type", "racecourse_track"]).reset_index(drop=True)
            years = track_df["work_date"].fillna("").astype(str).str[:4].replace("", "unknown")
            track_df = track_df.assign(_year=years)
            for year, year_df in track_df.groupby("_year", dropna=False):
                out_path = trackwork_output(str(year))
                header = track_headers.get(str(year), not out_path.exists())
                year_df = year_df.drop(columns=["_year"])
                year_df.to_csv(out_path, index=False, mode="a", header=header)
                track_headers[str(year)] = False
                total_track += len(year_df)

        vet_df = flatten(results, "vet")
        if "vet" in selected_pages and not vet_df.empty:
            vet_df = vet_df.sort_values(["horse_id", "record_date"]).reset_index(drop=True)
            vet_df.to_csv(VET_OUTPUT, index=False, mode="a", header=vet_header)
            vet_header = False
            total_vet += len(vet_df)
        batch_parts = []
        total_parts = []
        if "movement" in selected_pages:
            batch_parts.append(f"movement={len(move_df):,}")
            total_parts.append(f"movement={total_move:,}")
        if "trackwork" in selected_pages:
            batch_parts.append(f"trackwork={len(track_df):,}")
            total_parts.append(f"trackwork={total_track:,}")
        if "vet" in selected_pages:
            batch_parts.append(f"vet={len(vet_df):,}")
            total_parts.append(f"vet={total_vet:,}")
        log(f"Appended state batch: {', '.join(batch_parts)}; totals {', '.join(total_parts)}")

        if failures:
            with open(failure_path, "a", encoding="utf-8") as fh:
                for item, reason in failures:
                    fh.write(f"{item}\t{reason}\n")

    if "movement" in selected_pages:
        log(f"Wrote {total_move:,} movement rows -> {MOVE_OUTPUT}")
    if "trackwork" in selected_pages:
        log(f"Wrote {total_track:,} trackwork rows -> {TRACKWORK_OUTPUT}")
    if "vet" in selected_pages:
        log(f"Wrote {total_vet:,} horse vet rows -> {VET_OUTPUT}")

    if all_failures:
        log(f"Recorded {len(all_failures)} failures -> {failure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
