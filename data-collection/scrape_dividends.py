from __future__ import annotations

import argparse
import concurrent.futures
import threading
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "historical-data" / "aspx-results"
DIVIDENDS_DIR = ROOT / "data" / "historical-data" / "dividends"
BASE_URL = "https://racing.hkjc.com/en-us/local/information/localresults"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def normalize_space(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\xa0", " ").split())


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


def build_results_url(date_iso: str, racecourse: str, race_number: int) -> str:
    year, month, day = date_iso.split("-")
    params = {
        "RaceDate": f"{year}/{month}/{day}",
        "Racecourse": racecourse,
        "RaceNo": int(race_number),
    }
    return f"{BASE_URL}?{urlencode(params)}"


def load_race_inventory(year: int) -> pd.DataFrame:
    path = RESULTS_DIR / f"aspx-results-{year}.csv"
    df = pd.read_csv(path, dtype=str, usecols=lambda c: c in {"race_id", "Date", "race_number", "Track"})
    df = df.dropna(subset=["race_id", "Date", "race_number", "Track"]).copy()
    return df.drop_duplicates(subset=["race_id"]).sort_values(["Date", "race_number"])


def parse_dividends(html: str, race_id: str, date_iso: str, race_number: int, source_url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("div.dividend_tab table")
    if table is None:
        return []

    body = table.find("tbody") or table
    rows = body.find_all("tr")
    current_pool = None
    parsed: list[dict[str, object]] = []
    for row in rows:
        cells = row.find_all("td")
        if not cells:
            continue
        texts = [normalize_space(cell.get_text(" ", strip=True)) for cell in cells]
        if len(cells) >= 3 and cells[0].get("rowspan"):
            current_pool = texts[0]
            combination = texts[1]
            dividend = texts[2]
        elif len(texts) >= 2:
            combination = texts[0]
            dividend = texts[1]
        else:
            continue
        if not current_pool or not combination or not dividend:
            continue
        parsed.append({
            "race_id": race_id,
            "date": date_iso,
            "race_number": int(race_number),
            "pool": current_pool,
            "combination": combination,
            "dividend": dividend,
            "source_url": source_url,
        })
    return parsed


def scrape_race(row: tuple[str, str, str, str]) -> list[dict[str, object]]:
    race_id, date_iso, race_number, racecourse = row
    url = build_results_url(date_iso, racecourse, int(race_number))
    html = fetch_html(url)
    return parse_dividends(html, race_id, date_iso, int(race_number), url)


def run_year(year: int, workers: int, overwrite: bool) -> Path:
    inventory = load_race_inventory(year)
    output_path = DIVIDENDS_DIR / f"dividends-{year}.csv"
    DIVIDENDS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    rows: list[dict[str, object]] = []
    tasks = [tuple(rec) for rec in inventory[["race_id", "Date", "race_number", "Track"]].itertuples(index=False, name=None)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(scrape_race, task): task for task in tasks}
        done = 0
        total = len(future_map)
        for future in concurrent.futures.as_completed(future_map):
            task = future_map[future]
            done += 1
            try:
                rows.extend(future.result())
            except Exception as exc:
                log(f"[error] {task[0]}: {exc}")
            if done % 50 == 0 or done == total:
                log(f"[{year}] processed {done}/{total} races; rows={len(rows)}")

    df = pd.DataFrame(rows, columns=["race_id", "date", "race_number", "pool", "combination", "dividend", "source_url"])
    df = df.sort_values(["race_id", "pool", "combination"]).reset_index(drop=True) if not df.empty else df
    df.to_csv(output_path, index=False)
    log(f"[{year}] wrote {len(df):,} dividend rows -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape HKJC dividends into flat yearly CSVs.")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for year in sorted(set(args.years)):
        run_year(year, args.workers, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
