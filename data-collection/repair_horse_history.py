from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "historical-data"
RESULTS_DIR = DATA_DIR / "aspx-results"
HORSES_DIR = DATA_DIR / "horses"
DEFAULT_OUTPUT = HORSES_DIR / "horses_HK_2025_2026_repair.json"
BASE_URL = "https://racing.hkjc.com/en-us/local/information/horse"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PRINT_LOCK = threading.Lock()
HEADER_MAP = {
    "RC /Track/ Course": "RC/Track/ Course",
}


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def normalize_space(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def parse_date_ddmmyyyy(value: str) -> str:
    text = normalize_space(value)
    parts = text.split("/")
    if len(parts) != 3:
        return text
    day, month, year = parts
    if len(year) == 2:
        year = f"20{year}" if int(year) < 50 else f"19{year}"
    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"


def extract_entity_id(url: str | None, key: str) -> str | None:
    if not url:
        return None
    match = re.search(fr"{key}=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def horse_url(horse_id: str) -> str:
    return f"{BASE_URL}?{urlencode({'horseid': horse_id, 'Option': 1})}"


def fetch_html(url: str, retries: int = 3, timeout: int = 30) -> str:
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
            time.sleep(attempt)
    raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def locate_record_table(soup: BeautifulSoup):
    for table in soup.select("table.bigborder"):
        first_row = table.find("tr")
        if not first_row:
            continue
        headers = [normalize_space(td.get_text(" ", strip=True)).replace("\n", " ") for td in first_row.find_all("td")]
        if any(header == "Race Index" for header in headers):
            return table
    return None


def parse_horse_records(html: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    table = locate_record_table(soup)
    if table is None:
        return []

    rows = table.find_all("tr")
    if not rows:
        return []

    header_cells = rows[0].find_all("td", recursive=False)
    headers: list[str] = []
    excluded_index = None
    for index, cell in enumerate(header_cells):
        header = normalize_space(cell.get_text(" ", strip=True)).replace("\n", " ")
        if header == "Video Replay":
            excluded_index = index
            continue
        if header:
            headers.append(header)

    records: list[dict[str, object]] = []
    for row in rows[1:]:
        direct_cells = row.find_all("td", recursive=False)
        if not direct_cells:
            continue
        if any(cell.has_attr("colspan") for cell in direct_cells):
            continue

        record: dict[str, object] = {}
        header_index = 0
        for cell_index, cell in enumerate(direct_cells):
            if excluded_index is not None and cell_index == excluded_index:
                continue
            if header_index >= len(headers):
                break
            header = HEADER_MAP.get(headers[header_index], headers[header_index])
            header_index += 1

            text = normalize_space(cell.get_text(" ", strip=True))
            record[header] = None if text == "--" else text

            anchor = cell.find("a", href=True)
            if anchor and not anchor["href"].lower().startswith("javascript:"):
                href = anchor["href"]
                if href.startswith("/"):
                    href = f"https://racing.hkjc.com{href}"
                record[f"{header}_link"] = href

        if record and any(value for value in record.values()):
            if isinstance(record.get("Date"), str):
                record["Date"] = parse_date_ddmmyyyy(record["Date"])
            trainer_link = record.get("Trainer_link")
            jockey_link = record.get("Jockey_link")
            if trainer_link:
                record["trainer_id"] = extract_entity_id(str(trainer_link), "trainerid")
            if jockey_link:
                record["jockey_id"] = extract_entity_id(str(jockey_link), "jockeyid")
            records.append(record)

    return records


def collect_target_horse_ids(years: list[int]) -> list[str]:
    ids: set[str] = set()
    for year in years:
        path = RESULTS_DIR / f"aspx-results-{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str, usecols=lambda c: c == "horse_id")
        ids.update(hid for hid in df["horse_id"].dropna().astype(str).tolist() if hid)
    return sorted(ids)


def scrape_one(horse_id: str) -> tuple[str, list[dict[str, object]]]:
    url = horse_url(horse_id)
    html = fetch_html(url)
    records = parse_horse_records(html)
    return horse_id, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair horse history for target result years using static HKJC horse pages.")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    horse_ids = collect_target_horse_ids(sorted(set(args.years)))
    if not horse_ids:
        raise SystemExit("No horse_ids found in target aspx-results files")

    ensure_dir = args.output.parent
    ensure_dir.mkdir(parents=True, exist_ok=True)

    scraped: dict[str, list[dict[str, object]]] = {}
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_map = {executor.submit(scrape_one, horse_id): horse_id for horse_id in horse_ids}
        done = 0
        total = len(future_map)
        for future in concurrent.futures.as_completed(future_map):
            horse_id = future_map[future]
            done += 1
            try:
                hid, records = future.result()
                if records:
                    scraped[hid] = records
                else:
                    failures.append(horse_id)
                if done % 50 == 0 or done == total:
                    log(f"Processed {done}/{total} horse pages; scraped={len(scraped)} failures={len(failures)}")
            except Exception as exc:
                failures.append(horse_id)
                log(f"[error] {horse_id}: {exc}")

    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(scraped, fh, indent=2)
    log(f"Wrote {len(scraped)} horse histories to {args.output}")

    if failures:
        failed_path = args.output.with_suffix(".failed.txt")
        failed_path.write_text("\n".join(sorted(failures)), encoding="utf-8")
        log(f"Recorded {len(failures)} failures to {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
