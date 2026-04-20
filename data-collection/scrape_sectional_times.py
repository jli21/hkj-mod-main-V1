from __future__ import annotations

import argparse
import concurrent.futures
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = ROOT / "data" / "historical-data" / "sectional-times"
DEFAULT_RESULTS_DIR = ROOT / "data" / "historical-data" / "aspx-results"
LEGACY_RESULTS_DIR = ROOT.parent / "hkjc-mod" / "data" / "historical-data" / "aspx-results"
BASE_URL = "https://racing.hkjc.com/en-us/local/information/displaysectionaltime"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
SECTION_COLUMNS = [
    "race_id",
    "Date",
    "race_number",
    "race_name",
    "race_class",
    "race_class_num",
    "rating_band",
    "distance_m",
    "racecourse",
    "track",
    "course",
    "going",
    "finish_place",
    "horse_no",
    "horse_id",
    "horse_name",
    "finish_time",
    "finish_time_sec",
    "section_index",
    "section_label",
    "running_position",
    "behind_leader",
    "sectional_time",
    "sectional_time_sec",
    "cumulative_time_sec",
    "horse_subsplit_1",
    "horse_subsplit_1_sec",
    "horse_subsplit_2",
    "horse_subsplit_2_sec",
    "race_sectional_time",
    "race_sectional_time_sec",
    "race_cumulative_time",
    "race_cumulative_time_sec",
    "race_subsplit_1",
    "race_subsplit_1_sec",
    "race_subsplit_2",
    "race_subsplit_2_sec",
    "source_url",
]
PRINT_LOCK = threading.Lock()


@dataclass(frozen=True)
class MeetingTask:
    year: int
    date_iso: str
    date_ddmmyyyy: str
    expected_race_numbers: tuple[int, ...]


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def normalize_space(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_time_to_seconds(value: str | None) -> float | None:
    text = normalize_space(value)
    if not text or text == "--":
        return None
    text = text.strip("()")
    try:
        if ":" in text:
            mins, rest = text.split(":", 1)
            secs, hund = rest.split(".", 1)
            return int(mins) * 60 + int(secs) + int(hund) / 100
        parts = text.split(".")
        if len(parts) == 2:
            secs, hund = parts
            return int(secs) + int(hund) / 100
        if len(parts) == 3:
            mins, secs, hund = parts
            return int(mins) * 60 + int(secs) + int(hund) / 100
    except ValueError:
        return None
    return None


def parse_date_to_iso(value: str) -> str:
    text = normalize_space(value)
    if not text:
        raise ValueError("Empty date")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    if re.fullmatch(r"\d{2}/\d{2}/\d{4}", text):
        day, month, year = text.split("/")
        return f"{year}-{month}-{day}"
    raise ValueError(f"Unsupported date format: {value}")


def iso_to_ddmmyyyy(value: str) -> str:
    year, month, day = value.split("-")
    return f"{day}/{month}/{year}"


def make_race_id(date_iso: str, race_number: int) -> str:
    return f"{date_iso.replace('-', '')}{int(race_number):02d}"


def extract_query_id(url: str | None, key: str) -> str:
    if not url:
        return ""
    parsed = urlparse(urljoin(BASE_URL, url))
    return parse_qs(parsed.query).get(key, [""])[0]


def detect_results_dir(candidate: Path | None = None) -> Path:
    if candidate and candidate.exists():
        return candidate
    if DEFAULT_RESULTS_DIR.exists():
        return DEFAULT_RESULTS_DIR
    if LEGACY_RESULTS_DIR.exists():
        return LEGACY_RESULTS_DIR
    raise FileNotFoundError(
        "No aspx-results directory found. Pass --results-dir explicitly or provide --meeting-date."
    )


def fetch_html(url: str, timeout: int = 30, retries: int = 3, backoff: float = 1.5) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=timeout) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                return response.read().decode(charset, errors="replace")
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def parse_race_conditions(text: str) -> dict[str, object]:
    line = normalize_space(text)
    tokens = [token.strip() for token in line.split(" - ") if token.strip()]
    race_class = tokens[0] if tokens else ""
    race_class_num = None
    rating_band = ""
    distance_m = None
    track = ""
    course = ""
    going = tokens[-1] if len(tokens) >= 2 else ""

    class_match = re.search(r"Class\s+(\d+)", race_class, flags=re.IGNORECASE)
    if class_match:
        race_class_num = int(class_match.group(1))

    for token in tokens:
        dist_match = re.fullmatch(r"(\d+)M", token, flags=re.IGNORECASE)
        if dist_match:
            distance_m = int(dist_match.group(1))
            continue
        if re.fullmatch(r"\(.+\)", token):
            rating_band = token
            continue
        if token.upper() in {"TURF", "AWT"}:
            track = token
            continue
        course_match = re.search(r'"([^"]+)"\s+Course', token, flags=re.IGNORECASE)
        if course_match:
            course = course_match.group(1)

    return {
        "race_class": race_class,
        "race_class_num": race_class_num,
        "rating_band": rating_band,
        "distance_m": distance_m,
        "track": track,
        "course": course,
        "going": going,
    }


def parse_section_summary_cell(cell) -> dict[str, object]:
    strings = [normalize_space(s) for s in cell.stripped_strings if normalize_space(s)]
    if not strings:
        return {
            "main": "",
            "main_sec": None,
            "sub1": "",
            "sub1_sec": None,
            "sub2": "",
            "sub2_sec": None,
        }
    main = strings[0].strip("()")
    sub1 = strings[1] if len(strings) > 1 else ""
    sub2 = strings[2] if len(strings) > 2 else ""
    return {
        "main": main,
        "main_sec": parse_time_to_seconds(main),
        "sub1": sub1,
        "sub1_sec": parse_time_to_seconds(sub1),
        "sub2": sub2,
        "sub2_sec": parse_time_to_seconds(sub2),
    }


def parse_section_cell(cell) -> dict[str, object] | None:
    paragraphs = cell.find_all("p")
    if not paragraphs:
        return None
    position_p = paragraphs[0]
    running_position = normalize_space(position_p.find("span").get_text(" ", strip=True) if position_p.find("span") else "")
    behind_leader = normalize_space(position_p.find("i").get_text(" ", strip=True) if position_p.find("i") else "")
    time_p = paragraphs[1] if len(paragraphs) > 1 else None
    if time_p is None:
        return None
    parsed_time = parse_section_summary_cell(time_p)
    if not running_position and not behind_leader and not parsed_time["main"]:
        return None
    return {
        "running_position": int(running_position) if running_position.isdigit() else None,
        "behind_leader": behind_leader,
        "sectional_time": parsed_time["main"],
        "sectional_time_sec": parsed_time["main_sec"],
        "horse_subsplit_1": parsed_time["sub1"],
        "horse_subsplit_1_sec": parsed_time["sub1_sec"],
        "horse_subsplit_2": parsed_time["sub2"],
        "horse_subsplit_2_sec": parsed_time["sub2_sec"],
    }


def parse_meeting_metadata(soup: BeautifulSoup, requested_date: str) -> dict[str, str]:
    header = soup.select_one("div.search span.f_fl")
    text = normalize_space(header.get_text(" ", strip=True) if header else "")
    if not text:
        return {"meeting_date": requested_date, "racecourse": ""}
    match = re.search(r"Meeting Date:(\d{2}/\d{2}/\d{4}),\s*(.+)$", text)
    if not match:
        return {"meeting_date": requested_date, "racecourse": ""}
    meeting_date = match.group(1)
    venue = match.group(2)
    if venue.upper().startswith("SHA TIN"):
        racecourse = "ST"
    elif venue.upper().startswith("HAPPY VALLEY"):
        racecourse = "HV"
    else:
        racecourse = venue
    return {"meeting_date": meeting_date, "racecourse": racecourse}


def parse_race_div(race_div, meeting_meta: dict[str, str], source_url: str) -> list[dict[str, object]]:
    title_p = race_div.select_one("div.Race > p.bg_blue")
    if title_p is None:
        return []
    title_text = normalize_space(title_p.get_text(" ", strip=True))
    race_match = re.search(r"Race\s+(\d+)", title_text, flags=re.IGNORECASE)
    if not race_match:
        return []
    race_number = int(race_match.group(1))
    date_iso = parse_date_to_iso(meeting_meta["meeting_date"])
    race_id = make_race_id(date_iso, race_number)

    race_block = race_div.select_one("div.Race")
    spans = race_block.select("p.f_fl span") if race_block else []
    meta_line = normalize_space(spans[0].get_text(" ", strip=True) if spans else "")
    race_name = normalize_space(spans[1].get_text(" ", strip=True) if len(spans) > 1 else "")
    race_info = parse_race_conditions(meta_line)

    summary_table = race_block.find("table") if race_block else None
    race_cumulative = []
    race_sectionals = []
    if summary_table is not None:
        rows = summary_table.find_all("tr")
        if len(rows) >= 2:
            race_cumulative = [parse_section_summary_cell(cell) for cell in rows[0].find_all("td")[1:]]
            race_sectionals = [parse_section_summary_cell(cell) for cell in rows[1].find_all("td")[1:]]

    horse_table = race_div.select_one("table.race_table tbody")
    if horse_table is None:
        return []

    rows_out: list[dict[str, object]] = []
    for horse_row in horse_table.find_all("tr", recursive=False):
        cells = horse_row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue

        finish_place = normalize_space(cells[0].get_text(" ", strip=True))
        horse_no = normalize_space(cells[1].get_text(" ", strip=True))
        horse_anchor = cells[2].find("a")
        horse_text = normalize_space(horse_anchor.get_text(" ", strip=True) if horse_anchor else cells[2].get_text(" ", strip=True))
        horse_name = re.sub(r"\s*\([A-Z0-9]+\)$", "", horse_text).strip()
        horse_id = extract_query_id(horse_anchor.get("href", "") if horse_anchor else "", "horseid")
        finish_time = normalize_space(cells[-1].get_text(" ", strip=True))
        finish_time_sec = parse_time_to_seconds(finish_time)

        section_cells = cells[3:-1]
        cumulative_time_sec = 0.0
        for section_index, cell in enumerate(section_cells, start=1):
            parsed = parse_section_cell(cell)
            if parsed is None or parsed["sectional_time_sec"] is None:
                continue
            cumulative_time_sec += float(parsed["sectional_time_sec"])
            race_section = race_sectionals[section_index - 1] if section_index - 1 < len(race_sectionals) else {}
            race_cum = race_cumulative[section_index - 1] if section_index - 1 < len(race_cumulative) else {}

            rows_out.append({
                "race_id": race_id,
                "Date": date_iso,
                "race_number": race_number,
                "race_name": race_name,
                "race_class": race_info["race_class"],
                "race_class_num": race_info["race_class_num"],
                "rating_band": race_info["rating_band"],
                "distance_m": race_info["distance_m"],
                "racecourse": meeting_meta["racecourse"],
                "track": race_info["track"],
                "course": race_info["course"],
                "going": race_info["going"],
                "finish_place": int(finish_place) if finish_place.isdigit() else finish_place,
                "horse_no": int(horse_no) if horse_no.isdigit() else None,
                "horse_id": horse_id,
                "horse_name": horse_name,
                "finish_time": finish_time,
                "finish_time_sec": finish_time_sec,
                "section_index": section_index,
                "section_label": f"{section_index}st Sec." if section_index == 1 else f"{section_index}nd Sec." if section_index == 2 else f"{section_index}rd Sec." if section_index == 3 else f"{section_index}th Sec.",
                "running_position": parsed["running_position"],
                "behind_leader": parsed["behind_leader"],
                "sectional_time": parsed["sectional_time"],
                "sectional_time_sec": parsed["sectional_time_sec"],
                "cumulative_time_sec": round(cumulative_time_sec, 2),
                "horse_subsplit_1": parsed["horse_subsplit_1"],
                "horse_subsplit_1_sec": parsed["horse_subsplit_1_sec"],
                "horse_subsplit_2": parsed["horse_subsplit_2"],
                "horse_subsplit_2_sec": parsed["horse_subsplit_2_sec"],
                "race_sectional_time": race_section.get("main", ""),
                "race_sectional_time_sec": race_section.get("main_sec"),
                "race_cumulative_time": race_cum.get("main", ""),
                "race_cumulative_time_sec": race_cum.get("main_sec"),
                "race_subsplit_1": race_section.get("sub1", ""),
                "race_subsplit_1_sec": race_section.get("sub1_sec"),
                "race_subsplit_2": race_section.get("sub2", ""),
                "race_subsplit_2_sec": race_section.get("sub2_sec"),
                "source_url": source_url,
            })

    return rows_out


def parse_meeting_page(html: str, requested_date_ddmmyyyy: str, source_url: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    meeting_meta = parse_meeting_metadata(soup, requested_date_ddmmyyyy)
    race_divs = soup.find_all("div", id=re.compile(r"^Race\d+$"))
    rows: list[dict[str, object]] = []
    for race_div in race_divs:
        rows.extend(parse_race_div(race_div, meeting_meta, source_url))
    if not rows:
        return pd.DataFrame(columns=SECTION_COLUMNS)
    df = pd.DataFrame(rows)
    for column in SECTION_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA
    return df[SECTION_COLUMNS].sort_values(["race_id", "horse_no", "section_index"]).reset_index(drop=True)


def build_all_races_url(date_ddmmyyyy: str) -> str:
    return f"{BASE_URL}?{urlencode({'racedate': date_ddmmyyyy, 'All': 'True'})}"


def load_inventory_for_year(year: int, results_dir: Path) -> list[MeetingTask]:
    path = results_dir / f"aspx-results-{year}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing results inventory: {path}")

    df = pd.read_csv(path, dtype=str, usecols=lambda c: c in {"Date", "race_number", "race_id"})
    if "Date" not in df.columns or "race_number" not in df.columns:
        raise ValueError(f"{path} must include Date and race_number columns")

    df = df.dropna(subset=["Date", "race_number"]).copy()
    df["Date"] = df["Date"].map(parse_date_to_iso)
    df["race_number"] = pd.to_numeric(df["race_number"], errors="coerce")
    df = df.dropna(subset=["race_number"])
    df["race_number"] = df["race_number"].astype(int)

    tasks = []
    grouped = df.groupby("Date", sort=True)["race_number"]
    for date_iso, race_numbers in grouped:
        expected = tuple(sorted(set(int(num) for num in race_numbers.tolist())))
        tasks.append(MeetingTask(year=year, date_iso=date_iso, date_ddmmyyyy=iso_to_ddmmyyyy(date_iso), expected_race_numbers=expected))
    return tasks


def scrape_meeting(task: MeetingTask, sleep_seconds: float = 0.0) -> pd.DataFrame:
    url = build_all_races_url(task.date_ddmmyyyy)
    html = fetch_html(url)
    df = parse_meeting_page(html, task.date_ddmmyyyy, url)
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    if df.empty:
        raise RuntimeError(f"No sectional rows parsed for {task.date_ddmmyyyy}")

    found_races = set(df["race_number"].dropna().astype(int).unique().tolist())
    expected_races = set(task.expected_race_numbers)
    missing = sorted(expected_races - found_races)
    if missing:
        log(f"[warn] {task.date_ddmmyyyy}: missing races from sectional page -> {missing}")
    return df[df["race_number"].isin(expected_races)].copy() if expected_races else df


def run_year(year: int, results_dir: Path, output_dir: Path, workers: int, sleep_seconds: float, overwrite: bool) -> Path:
    tasks = load_inventory_for_year(year, results_dir)
    if not tasks:
        raise RuntimeError(f"No meeting dates found for {year}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sectional_times_{year}.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")

    log(f"[{year}] scraping {len(tasks)} meetings with {workers} workers")
    dfs: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(scrape_meeting, task, sleep_seconds): task for task in tasks}
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            try:
                df = future.result()
                dfs.append(df)
                log(f"[{year}] {task.date_ddmmyyyy}: {len(df):,} rows")
            except Exception as exc:
                log(f"[error] [{year}] {task.date_ddmmyyyy}: {exc}")

    dfs = [df for df in dfs if not df.empty]
    if not dfs:
        empty_df = pd.DataFrame(columns=SECTION_COLUMNS)
        empty_df.to_csv(output_path, index=False)
        log(f"[{year}] no published sectional data found; wrote header-only file -> {output_path}")
        return output_path

    year_df = pd.concat(dfs, ignore_index=True)
    year_df = year_df.sort_values(["race_id", "horse_no", "section_index"]).reset_index(drop=True)
    year_df.to_csv(output_path, index=False)
    log(f"[{year}] wrote {len(year_df):,} rows -> {output_path}")
    return output_path


def run_single_meeting(meeting_date: str, output_dir: Path, overwrite: bool) -> Path:
    date_iso = parse_date_to_iso(meeting_date)
    task = MeetingTask(
        year=int(date_iso[:4]),
        date_iso=date_iso,
        date_ddmmyyyy=iso_to_ddmmyyyy(date_iso),
        expected_race_numbers=tuple(),
    )
    df = scrape_meeting(task)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"sectional_times_{task.year}.csv"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite.")
    df.to_csv(output_path, index=False)
    log(f"[single] wrote {len(df):,} rows -> {output_path}")
    return output_path


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scrape HKJC sectional times into one yearly CSV. "
            "Each row is one horse-section, with race-level summary fields repeated."
        )
    )
    parser.add_argument("--years", nargs="+", type=int, help="Explicit years to scrape, e.g. --years 2019 2020")
    parser.add_argument("--start-year", type=int, default=None, help="Range start year")
    parser.add_argument("--end-year", type=int, default=None, help="Range end year")
    parser.add_argument("--meeting-date", type=str, help="Single meeting date in YYYY-MM-DD or DD/MM/YYYY")
    parser.add_argument("--results-dir", type=Path, default=None, help="Directory containing aspx-results-<year>.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for sectional_times_<year>.csv")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent meeting fetches")
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep after each meeting fetch")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSVs")
    return parser.parse_args(list(argv))


def resolve_years(args: argparse.Namespace) -> list[int]:
    if args.years:
        return sorted(set(args.years))
    if args.start_year is not None and args.end_year is not None:
        return list(range(args.start_year, args.end_year + 1))
    return []


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.meeting_date:
        run_single_meeting(args.meeting_date, args.output_dir, args.overwrite)
        return 0

    years = resolve_years(args)
    if not years:
        raise SystemExit("Provide --meeting-date, --years, or --start-year/--end-year")

    results_dir = detect_results_dir(args.results_dir)
    failures = []
    for year in years:
        try:
            run_year(
                year=year,
                results_dir=results_dir,
                output_dir=args.output_dir,
                workers=max(1, args.workers),
                sleep_seconds=max(0.0, args.sleep),
                overwrite=args.overwrite,
            )
        except Exception as exc:
            failures.append((year, str(exc)))
            log(f"[error] [{year}] {exc}")

    if failures:
        for year, message in failures:
            log(f"[failed] {year}: {message}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
