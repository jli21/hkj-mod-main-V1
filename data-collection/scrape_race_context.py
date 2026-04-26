from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

from scrape_hkjc_common import (
    DATA_DIR,
    ensure_dir,
    fetch_html,
    load_race_index,
    load_runner_index,
    log,
    make_race_id,
    meeting_index_from_races,
    merge_runner_ids,
    normalize_space,
    parse_ddmmyyyy,
    run_parallel,
    scrape_timestamp,
)


COMMENTS_DIR = DATA_DIR / "comments-on-running"
REPORTS_DIR = DATA_DIR / "race-incidents"
EXCEPTIONAL_DIR = DATA_DIR / "race-exceptional-factors"
VET_DIR = DATA_DIR / "race-veterinary-records"
FORMLINE_DIR = DATA_DIR / "race-formline"
HORSES_DIR = DATA_DIR / "horses"
HORSE_VET_PATH = DATA_DIR / "horse-veterinary-records" / "horse-veterinary-records.csv"
HORSE_MOVEMENT_PATH = DATA_DIR / "horse-movement-records" / "horse-movement-records.csv"

PAGE_CHOICES = ["comments", "reports", "exceptional", "vet", "formline"]


def comments_url(date_iso: str, race_number: int) -> str:
    return f"https://racing.hkjc.com/en-us/local/information/corunning?date={date_iso.replace('-', '')}&raceno={race_number}"


def report_url(date_iso: str, racecourse: str) -> str:
    date_path = date_iso.replace("-", "/")
    return f"https://racing.hkjc.com/en-us/local/information/racereportfull?racedate={date_path}&Racecourse={racecourse}&RaceNo=1"


def exceptional_url(date_iso: str, racecourse: str, race_number: int) -> str:
    date_path = date_iso.replace("-", "/")
    return f"https://racing.hkjc.com/en-us/local/information/exceptionalfactors?racedate={date_path}&Racecourse={racecourse}&RaceNo={race_number}"


def vet_url(date_iso: str, racecourse: str, race_number: int) -> str:
    date_path = date_iso.replace("-", "/")
    return f"https://racing.hkjc.com/en-us/local/information/veterinaryrecord?racedate={date_path}&Racecourse={racecourse}&RaceNo={race_number}"


def formline_url(date_iso: str, racecourse: str, race_number: int) -> str:
    date_path = date_iso.replace("-", "/")
    return f"https://racing.hkjc.com/en-us/local/information/formline?racedate={date_path}&Racecourse={racecourse}&RaceNo={race_number}"


def table_headers(table) -> list[str]:
    first_row = table.find("tr")
    if first_row is None:
        return []
    return [normalize_space(cell.get_text(" ", strip=True)) for cell in first_row.find_all(["td", "th"])]


def extract_horse_from_cell(cell) -> tuple[str | None, str | None]:
    text = normalize_space(cell.get_text(" ", strip=True))
    horse_id = None
    anchor = cell.find("a", href=True)
    if anchor is not None:
        href = anchor.get("href", "")
        match = re.search(r"horseid=([A-Za-z0-9_]+)", href, flags=re.IGNORECASE)
        if match:
            horse_id = match.group(1)
    horse_name = re.sub(r"\s*\([A-Z0-9]+\)$", "", text).strip() or None
    return horse_name, horse_id


def parse_comments_page(race: dict[str, object], html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    scrape_ts = scrape_timestamp()
    for table in soup.find_all("table"):
        if table_headers(table)[:6] != ["Placing", "Horse No", "Horse Name", "Jockey", "Gear", "Comment"]:
            continue
        rows: list[dict[str, object]] = []
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 6:
                continue
            horse_name, horse_id = extract_horse_from_cell(cells[2])
            rows.append(
                {
                    "race_id": race["race_id"],
                    "Date": race["Date"],
                    "race_number": race["race_number"],
                    "Racecourse": race["Track"],
                    "Placing": normalize_space(cells[0].get_text(" ", strip=True)) or None,
                    "Horse No.": normalize_space(cells[1].get_text(" ", strip=True)) or None,
                    "horse_name": horse_name,
                    "horse_id": horse_id,
                    "Jockey": normalize_space(cells[3].get_text(" ", strip=True)) or None,
                    "Gear": normalize_space(cells[4].get_text(" ", strip=True)) or None,
                    "Comment": normalize_space(cells[5].get_text(" ", strip=True)) or None,
                    "source_url": comments_url(str(race["Date"]), int(race["race_number"])),
                    "scrape_ts": scrape_ts,
                }
            )
        return rows
    return []


def parse_report_meeting(meeting: dict[str, object], html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    scrape_ts = scrape_timestamp()
    race_tables = [
        table
        for table in soup.find_all("table")
        if table_headers(table)[:7] == ["Pla.", "Horse No", "Colour", "Horse", "Dr.", "Jockey", "Incident"]
    ]
    rows: list[dict[str, object]] = []
    for race_number, table in enumerate(race_tables, start=1):
        race_id = make_race_id(str(meeting["Date"]), race_number)
        for tr in table.find_all("tr")[1:]:
            cells = tr.find_all("td")
            if len(cells) < 7:
                continue
            horse_name, horse_id = extract_horse_from_cell(cells[3])
            incident = normalize_space(cells[6].get_text(" ", strip=True)) or None
            rows.append(
                {
                    "race_id": race_id,
                    "Date": meeting["Date"],
                    "race_number": race_number,
                    "Racecourse": meeting["Racecourse"],
                    "Placing": normalize_space(cells[0].get_text(" ", strip=True)) or None,
                    "Horse No.": normalize_space(cells[1].get_text(" ", strip=True)) or None,
                    "horse_name": horse_name,
                    "horse_id": horse_id,
                    "Draw": normalize_space(cells[4].get_text(" ", strip=True)) or None,
                    "Jockey": normalize_space(cells[5].get_text(" ", strip=True)) or None,
                    "Incident": incident,
                    "requires_trial": bool(incident and "barrier trial" in incident.lower()),
                    "requires_vet_exam": bool(incident and "official veterinary examination" in incident.lower()),
                    "sampling_flag": bool(incident and "sampling post-race" in incident.lower()),
                    "additional_vet_report": bool(incident and "additional veterinary report" in incident.lower()),
                    "source_url": report_url(str(meeting["Date"]), str(meeting["Racecourse"])),
                    "scrape_ts": scrape_ts,
                }
            )
    return rows


def parse_exceptional_page(race: dict[str, object], html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    scrape_ts = scrape_timestamp()
    expected = [
        "Horse No.", "Horse", "Out of the Handicap", "WFA", "Gear", "Tongue Tie",
        "Bit Change From", "Bit Change To", "Rtg.", "Dist", "Days since Last Run",
        "New Trainer", "Date returned from Conghua",
    ]
    for table in soup.find_all("table"):
        if table_headers(table)[: len(expected)] != expected:
            continue
        rows: list[dict[str, object]] = []
        for tr in table.find_all("tr")[1:]:
            cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
            if not cells:
                continue
            cells += [""] * (len(expected) - len(cells))
            rows.append(
                {
                    "race_id": race["race_id"],
                    "Date": race["Date"],
                    "race_number": race["race_number"],
                    "Racecourse": race["Track"],
                    "Horse No.": cells[0] or None,
                    "Horse": cells[1] or None,
                    "Out of the Handicap": cells[2] or None,
                    "WFA": cells[3] or None,
                    "Gear": cells[4] or None,
                    "Tongue Tie": cells[5] or None,
                    "Bit Change From": cells[6] or None,
                    "Bit Change To": cells[7] or None,
                    "Rtg.": cells[8] or None,
                    "Dist": cells[9] or None,
                    "Days since Last Run": cells[10] or None,
                    "New Trainer": cells[11] or None,
                    "Date returned from Conghua": parse_ddmmyyyy(cells[12]),
                    "source_url": exceptional_url(str(race["Date"]), str(race["Track"]), int(race["race_number"])),
                    "scrape_ts": scrape_ts,
                }
            )
        return rows
    return []


def parse_race_vet_page(race: dict[str, object], html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    scrape_ts = scrape_timestamp()
    expected = ["Horse No.", "Horse Name", "Date", "Details", "Passed On"]
    for table in soup.find_all("table"):
        if table_headers(table)[:5] != expected:
            continue
        rows: list[dict[str, object]] = []
        current_no = None
        current_name = None
        for tr in table.find_all("tr")[1:]:
            cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
            if not cells:
                continue
            if len(cells) >= 5:
                current_no, current_name = cells[0] or None, cells[1] or None
                date_val, details, passed = cells[2], cells[3], cells[4]
            elif len(cells) == 3:
                date_val, details, passed = cells[0], cells[1], cells[2]
            else:
                continue
            rows.append(
                {
                    "race_id": race["race_id"],
                    "Date": race["Date"],
                    "race_number": race["race_number"],
                    "Racecourse": race["Track"],
                    "Horse No.": current_no,
                    "Horse": current_name,
                    "record_date": parse_ddmmyyyy(date_val),
                    "Details": details or None,
                    "Passed On": parse_ddmmyyyy(passed),
                    "source_url": vet_url(str(race["Date"]), str(race["Track"]), int(race["race_number"])),
                    "scrape_ts": scrape_ts,
                }
            )
        return rows
    return []


def parse_formline_page(race: dict[str, object], html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    scrape_ts = scrape_timestamp()
    rows: list[dict[str, object]] = []
    for table in soup.find_all("table"):
        parsed = []
        for tr in table.find_all("tr"):
            cells = [normalize_space(td.get_text(" ", strip=True)) for td in tr.find_all("td")]
            if cells:
                parsed.append(cells)
        if len(parsed) < 3:
            continue
        if not re.match(r"\d{2}/\d{2}/\d{4} \(\d+\)", parsed[0][0]):
            continue
        reference_match = re.match(r"(\d{2}/\d{2}/\d{4}) \((\d+)\)", parsed[0][0])
        if not reference_match:
            continue
        headers = parsed[1]
        data_rows = parsed[2:]
        meta = parsed[0] + [""] * 6
        ref_date = parse_ddmmyyyy(reference_match.group(1))
        ref_race_index = reference_match.group(2)
        for data_row in data_rows:
            padded = data_row + [""] * (len(headers) - len(data_row))
            record = {
                "race_id": race["race_id"],
                "Date": race["Date"],
                "race_number": race["race_number"],
                "Racecourse": race["Track"],
                "reference_date": ref_date,
                "reference_race_index": ref_race_index,
                "reference_racecourse": meta[1] or None,
                "reference_distance": meta[2] or None,
                "reference_class": meta[3] or None,
                "reference_course": meta[4] or None,
                "reference_going": meta[5] or None,
                "source_url": formline_url(str(race["Date"]), str(race["Track"]), int(race["race_number"])),
                "scrape_ts": scrape_ts,
            }
            for header, value in zip(headers, padded):
                record[header] = value or None
            rows.append(record)
    return rows


def scrape_comments_worker(row: dict[str, object]) -> list[dict[str, object]]:
    return parse_comments_page(row, fetch_html(comments_url(str(row["Date"]), int(row["race_number"]))))


def scrape_exceptional_worker(row: dict[str, object]) -> list[dict[str, object]]:
    return parse_exceptional_page(row, fetch_html(exceptional_url(str(row["Date"]), str(row["Track"]), int(row["race_number"]))))


def scrape_vet_worker(row: dict[str, object]) -> list[dict[str, object]]:
    return parse_race_vet_page(row, fetch_html(vet_url(str(row["Date"]), str(row["Track"]), int(row["race_number"]))))


def scrape_formline_worker(row: dict[str, object]) -> list[dict[str, object]]:
    return parse_formline_page(row, fetch_html(formline_url(str(row["Date"]), str(row["Track"]), int(row["race_number"]))))


def scrape_report_worker(meeting: dict[str, object]) -> list[dict[str, object]]:
    return parse_report_meeting(meeting, fetch_html(report_url(str(meeting["Date"]), str(meeting["Racecourse"]))))


def flatten(results: list[list[dict[str, object]]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for result in results:
        rows.extend(result)
    return pd.DataFrame(rows)


def maybe_float(value: object) -> float | None:
    text = normalize_space(value)
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def clean_nullable(value: object) -> str | None:
    text = normalize_space(value)
    if not text or text.lower() in {"nan", "none", "nat"}:
        return None
    return text


def load_horse_history_for_ids(horse_ids: set[str]) -> dict[str, list[dict[str, object]]]:
    history: dict[str, list[dict[str, object]]] = {horse_id: [] for horse_id in horse_ids}
    for path in sorted(HORSES_DIR.glob("horses_HK_*.json")):
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        keys = horse_ids.intersection(payload.keys())
        for horse_id in keys:
            records = payload.get(horse_id, [])
            normalized: list[dict[str, object]] = []
            for record in records:
                record_copy = dict(record)
                record_copy["Date"] = parse_ddmmyyyy(record_copy.get("Date"))
                normalized.append(record_copy)
            history[horse_id] = normalized
    return history


def load_horse_vet_frame() -> pd.DataFrame:
    if not HORSE_VET_PATH.exists():
        return pd.DataFrame(columns=["horse_id", "record_date", "details", "passed_date"])
    df = pd.read_csv(HORSE_VET_PATH, dtype=str)
    for column in ["record_date", "passed_date"]:
        if column in df.columns:
            df[column] = df[column].astype(str)
    return df


def load_movement_frame() -> pd.DataFrame:
    if not HORSE_MOVEMENT_PATH.exists():
        return pd.DataFrame(columns=["horse_id", "from_location", "to_location", "arrival_date"])
    return pd.read_csv(HORSE_MOVEMENT_PATH, dtype=str)


def current_history_record(records: list[dict[str, object]], race_date: str) -> dict[str, object] | None:
    for record in records:
        if record.get("Date") == race_date:
            return record
    return None


def previous_history_record(records: list[dict[str, object]], race_date: str) -> dict[str, object] | None:
    previous = [record for record in records if normalize_space(record.get("Date")) and str(record.get("Date")) < race_date]
    previous.sort(key=lambda item: str(item.get("Date")), reverse=True)
    return previous[0] if previous else None


def infer_tongue_tie(gear: str | None) -> str | None:
    text = normalize_space(gear)
    if not text:
        return None
    if "TT" not in text:
        return None
    match = re.search(r"TT[12-]?", text)
    return match.group(0) if match else "TT"


def latest_return_from_conghua(movement_df: pd.DataFrame, horse_id: str, race_date: str) -> str | None:
    if movement_df.empty:
        return None
    subset = movement_df[
        (movement_df["horse_id"] == horse_id)
        & (movement_df["to_location"].fillna("").str.upper() == "HONG KONG")
        & (movement_df["arrival_date"].fillna("") <= race_date)
    ]
    if subset.empty:
        return None
    return clean_nullable(subset.sort_values("arrival_date").iloc[-1]["arrival_date"])


def build_exceptional_fallback(year: int, races: pd.DataFrame, runner_index: pd.DataFrame) -> pd.DataFrame:
    horse_ids = set(runner_index["horse_id"].dropna().astype(str).tolist())
    history = load_horse_history_for_ids(horse_ids)
    movement_df = load_movement_frame()
    scrape_ts = scrape_timestamp()
    rows: list[dict[str, object]] = []
    for race in races.to_dict("records"):
        race_id = str(race["race_id"])
        race_date = str(race["Date"])
        entries = runner_index[runner_index["race_id"] == race_id].sort_values("Horse No.")
        for _, entry in entries.iterrows():
            horse_id = str(entry["horse_id"])
            records = history.get(horse_id, [])
            current = current_history_record(records, race_date)
            previous = previous_history_record(records, race_date)
            current_trainer = normalize_space(current.get("Trainer")) if current else None
            previous_trainer = normalize_space(previous.get("Trainer")) if previous else None
            days_since_last = None
            if previous is not None and previous.get("Date"):
                try:
                    days_since_last = str((pd.Timestamp(race_date) - pd.Timestamp(str(previous.get("Date")))).days)
                except Exception:
                    days_since_last = None
            rows.append(
                {
                    "race_id": race_id,
                    "Date": race_date,
                    "race_number": race["race_number"],
                    "Racecourse": race["Track"],
                    "Horse No.": entry["Horse No."],
                    "Horse": normalize_space(entry["Horse"]),
                    "Out of the Handicap": None,
                    "WFA": None,
                    "Gear": normalize_space(current.get("Gear")) if current else None,
                    "Tongue Tie": infer_tongue_tie(current.get("Gear") if current else None),
                    "Bit Change From": None,
                    "Bit Change To": None,
                    "Rtg.": clean_nullable(current.get("Rtg.")) if current else None,
                    "Dist": clean_nullable(current.get("Dist.")) if current else None,
                    "Days since Last Run": days_since_last,
                    "New Trainer": "YES" if current_trainer and previous_trainer and current_trainer != previous_trainer else None,
                    "Date returned from Conghua": latest_return_from_conghua(movement_df, horse_id, race_date),
                    "horse_id": horse_id,
                    "source_url": None,
                    "scrape_ts": scrape_ts,
                }
            )
    return pd.DataFrame(rows)


def build_vet_fallback(year: int, races: pd.DataFrame, runner_index: pd.DataFrame) -> pd.DataFrame:
    horse_vet = load_horse_vet_frame()
    scrape_ts = scrape_timestamp()
    rows: list[dict[str, object]] = []
    if horse_vet.empty:
        return pd.DataFrame(rows)
    for race in races.to_dict("records"):
        race_id = str(race["race_id"])
        race_date = str(race["Date"])
        entries = runner_index[runner_index["race_id"] == race_id]
        for _, entry in entries.iterrows():
            horse_id = str(entry["horse_id"])
            subset = horse_vet[(horse_vet["horse_id"] == horse_id) & (horse_vet["record_date"].fillna("") <= race_date)]
            for _, record in subset.iterrows():
                rows.append(
                    {
                        "race_id": race_id,
                        "Date": race_date,
                        "race_number": race["race_number"],
                        "Racecourse": race["Track"],
                        "Horse No.": entry["Horse No."],
                        "Horse": normalize_space(entry["Horse"]),
                        "horse_id": horse_id,
                        "record_date": clean_nullable(record.get("record_date")),
                        "Details": clean_nullable(record.get("details")),
                        "Passed On": clean_nullable(record.get("passed_date")),
                        "source_url": None,
                        "scrape_ts": scrape_ts,
                    }
                )
    return pd.DataFrame(rows)


def build_formline_fallback(year: int, races: pd.DataFrame, runner_index: pd.DataFrame) -> pd.DataFrame:
    horse_ids = set(runner_index["horse_id"].dropna().astype(str).tolist())
    history = load_horse_history_for_ids(horse_ids)
    scrape_ts = scrape_timestamp()
    rows: list[dict[str, object]] = []
    for race in races.to_dict("records"):
        race_id = str(race["race_id"])
        race_date = str(race["Date"])
        entries = runner_index[runner_index["race_id"] == race_id]
        current_records: dict[str, dict[str, object]] = {}
        reference_groups: dict[tuple[str, str], list[tuple[pd.Series, dict[str, object], dict[str, object]]]] = {}
        for _, entry in entries.iterrows():
            horse_id = str(entry["horse_id"])
            records = history.get(horse_id, [])
            current = current_history_record(records, race_date)
            if current is None:
                continue
            current_records[horse_id] = current
            for record in records:
                record_date = str(record.get("Date") or "")
                if not record_date or record_date >= race_date:
                    continue
                key = (record_date, normalize_space(record.get("Race Index")))
                reference_groups.setdefault(key, []).append((entry, current, record))
        for (reference_date, reference_race_index), participants in reference_groups.items():
            if len(participants) < 2:
                continue
            sample_record = participants[0][2]
            rc_track_course = normalize_space(sample_record.get("RC/Track/ Course"))
            ref_parts = [normalize_space(part) for part in rc_track_course.split("/")]
            ref_course = ref_parts[2] if len(ref_parts) >= 3 else None
            ref_racecourse = ref_parts[0] if len(ref_parts) >= 1 else None
            for entry, current, previous in participants:
                current_wt = maybe_float(current.get("Act. Wt."))
                previous_wt = maybe_float(previous.get("Act. Wt."))
                current_rtg = maybe_float(current.get("Rtg."))
                previous_rtg = maybe_float(previous.get("Rtg."))
                rows.append(
                    {
                        "race_id": race_id,
                        "Date": race_date,
                        "race_number": race["race_number"],
                        "Racecourse": race["Track"],
                        "reference_date": reference_date,
                        "reference_race_index": reference_race_index,
                        "reference_racecourse": ref_racecourse,
                        "reference_distance": clean_nullable(previous.get("Dist.")),
                        "reference_class": clean_nullable(previous.get("Race Class")),
                        "reference_course": ref_course,
                        "reference_going": clean_nullable(previous.get("G")),
                        "Horse": normalize_space(entry["Horse"]),
                        "horse_id": entry["horse_id"],
                        "Wt.": clean_nullable(current.get("Act. Wt.")),
                        "Rtg.": clean_nullable(current.get("Rtg.")),
                        "Draw": clean_nullable(current.get("Dr.")),
                        "Jockey": clean_nullable(current.get("Jockey")),
                        "Last Jockey": clean_nullable(previous.get("Jockey")),
                        "Placing": clean_nullable(previous.get("Pla.")),
                        "Time": clean_nullable(previous.get("Finish Time")),
                        "Margin": clean_nullable(previous.get("LBW")),
                        "Win Odds": clean_nullable(previous.get("Win Odds")),
                        "Weight Carried +/-": None if current_wt is None or previous_wt is None else current_wt - previous_wt,
                        "Rating +/-": None if current_rtg is None or previous_rtg is None else current_rtg - previous_rtg,
                        "source_url": None,
                        "scrape_ts": scrape_ts,
                    }
                )
    return pd.DataFrame(rows)


def fallback_frame(page: str, year: int, races: pd.DataFrame, runner_index: pd.DataFrame) -> pd.DataFrame:
    if page == "exceptional":
        return build_exceptional_fallback(year, races, runner_index)
    if page == "vet":
        return build_vet_fallback(year, races, runner_index)
    if page == "formline":
        return build_formline_fallback(year, races, runner_index)
    return pd.DataFrame()


def page_output(page: str, year: int) -> Path:
    if page == "comments":
        return COMMENTS_DIR / f"comments-on-running-{year}.csv"
    if page == "reports":
        return REPORTS_DIR / f"race-incidents-{year}.csv"
    if page == "exceptional":
        return EXCEPTIONAL_DIR / f"exceptional-factors-{year}.csv"
    if page == "vet":
        return VET_DIR / f"race-veterinary-records-{year}.csv"
    if page == "formline":
        return FORMLINE_DIR / f"formline-{year}.csv"
    raise ValueError(page)


def page_columns(page: str) -> list[str]:
    if page == "comments":
        return [
            "race_id", "Date", "race_number", "Racecourse", "Placing", "Horse No.", "horse_name",
            "horse_id", "Jockey", "Gear", "Comment", "source_url", "scrape_ts",
        ]
    if page == "reports":
        return [
            "race_id", "Date", "race_number", "Racecourse", "Placing", "Horse No.", "horse_name",
            "horse_id", "Draw", "Jockey", "Incident", "requires_trial", "requires_vet_exam",
            "sampling_flag", "additional_vet_report", "source_url", "scrape_ts",
        ]
    if page == "exceptional":
        return [
            "race_id", "Date", "race_number", "Racecourse", "Horse No.", "Horse", "Out of the Handicap",
            "WFA", "Gear", "Tongue Tie", "Bit Change From", "Bit Change To", "Rtg.", "Dist",
            "Days since Last Run", "New Trainer", "Date returned from Conghua", "horse_id", "source_url",
            "scrape_ts",
        ]
    if page == "vet":
        return [
            "race_id", "Date", "race_number", "Racecourse", "Horse No.", "Horse", "horse_id",
            "record_date", "Details", "Passed On", "source_url", "scrape_ts",
        ]
    if page == "formline":
        return [
            "race_id", "Date", "race_number", "Racecourse", "reference_date", "reference_race_index",
            "reference_racecourse", "reference_distance", "reference_class", "reference_course",
            "reference_going", "Horse", "horse_id", "Wt.", "Rtg.", "Draw", "Jockey", "Last Jockey",
            "Placing", "Time", "Margin", "Win Odds", "Weight Carried +/-", "Rating +/-", "source_url",
            "scrape_ts",
        ]
    raise ValueError(page)


def scrape_page_for_year(page: str, year: int, workers: int, overwrite: bool, fallback_only: bool = False) -> Path:
    output = page_output(page, year)
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output}")
    ensure_dir(output.parent)

    races = load_race_index([year])
    runner_index = load_runner_index([year])
    if races.empty:
        raise RuntimeError(f"No races found for year {year}")

    if fallback_only and page not in {"exceptional", "vet", "formline"}:
        raise ValueError(f"fallback_only is only supported for exceptional/vet/formline, got {page}")

    if fallback_only:
        results = []
        failures = []
        df = fallback_frame(page, year, races, runner_index)
    elif page == "reports":
        items = meeting_index_from_races(races).to_dict("records")
        results, failures = run_parallel(items, scrape_report_worker, workers, f"reports-{year}", progress_every=25)
        df = flatten(results)
    else:
        items = races.to_dict("records")
        worker_map = {
            "comments": scrape_comments_worker,
            "exceptional": scrape_exceptional_worker,
            "vet": scrape_vet_worker,
            "formline": scrape_formline_worker,
        }
        results, failures = run_parallel(items, worker_map[page], workers, f"{page}-{year}", progress_every=50)
        df = flatten(results)
    if page in {"comments", "reports", "exceptional", "vet"}:
        df = merge_runner_ids(df, runner_index)
    if not fallback_only and df.empty and page in {"exceptional", "vet", "formline"}:
        log(f"[{page}-{year}] direct endpoint returned no rows; using derived fallback")
        df = fallback_frame(page, year, races, runner_index)
    if df.empty:
        df = pd.DataFrame(columns=page_columns(page))
    else:
        for column in page_columns(page):
            if column not in df.columns:
                df[column] = pd.NA
        df = df[page_columns(page) + [col for col in df.columns if col not in page_columns(page)]]
    if not df.empty and "Horse No." in df.columns:
        df["Horse No."] = df["Horse No."].astype(str)
    if not df.empty:
        sort_cols = [col for col in ["race_id", "Horse No.", "reference_date", "record_date"] if col in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True) if sort_cols else df
    df.to_csv(output, index=False)
    log(f"[{page}-{year}] wrote {len(df):,} rows -> {output}")

    if failures:
        failure_path = output.with_suffix(".failed.txt")
        failure_path.write_text("\n".join(f"{item}\t{reason}" for item, reason in failures), encoding="utf-8")
        log(f"[{page}-{year}] recorded {len(failures)} failures -> {failure_path}")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape HKJC race context pages into yearly CSVs.")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--pages", nargs="+", choices=PAGE_CHOICES, default=PAGE_CHOICES)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fallback-only", action="store_true")
    args = parser.parse_args()

    for page in args.pages:
        for year in sorted(set(args.years)):
            scrape_page_for_year(page, year, args.workers, args.overwrite, fallback_only=args.fallback_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
