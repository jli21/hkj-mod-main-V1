from __future__ import annotations

import argparse
import concurrent.futures
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
OUTPUT_DIR = ROOT / "data" / "historical-data" / "barrier-trial-results"
BASE_URL = "https://racing.hkjc.com/en-us/local/information/btresult"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PRINT_LOCK = threading.Lock()
RACE_HEADER_RE = re.compile(r"Batch\s*(\d+)\s*-\s*(.*?)\s*-\s*(\d+)[mM]")


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


def build_url(date_ddmmyyyy: str) -> str:
    return f"{BASE_URL}?{urlencode({'Date': date_ddmmyyyy})}"


def parse_date_iso(date_ddmmyyyy: str) -> str:
    day, month, year = date_ddmmyyyy.split("/")
    return f"{year}-{month}-{day}"


def make_race_id(date_iso: str, trial_number: int) -> str:
    return f"{date_iso.replace('-', '')}{trial_number:02d}"


def extract_horse_id(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"horseid=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def normalize_track(text: str) -> str:
    upper = text.upper()
    if "CONGHUA" in upper and "ALL WEATHER" in upper:
        return "CG_AWT"
    if "CONGHUA" in upper:
        return "CG"
    if "SHA TIN" in upper and "ALL WEATHER" in upper:
        return "ST_AWT"
    if "SHA TIN" in upper:
        return "ST"
    if "HAPPY VALLEY" in upper and "ALL WEATHER" in upper:
        return "HV_AWT"
    if "HAPPY VALLEY" in upper:
        return "HV"
    return text


def iter_year_dates(year: int) -> list[str]:
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    return [ts.strftime("%d/%m/%Y") for ts in dates]


def parse_trial_page(date_ddmmyyyy: str, html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    rows: list[dict[str, object]] = []
    date_iso = parse_date_iso(date_ddmmyyyy)

    for anchor in soup.select("a[id^='stb']"):
        subheader = anchor.find_next("td", class_=lambda c: c and "subheader" in c)
        if subheader is None:
            continue
        header_text = normalize_space(subheader.get_text(" ", strip=True))
        match = RACE_HEADER_RE.match(header_text)
        if not match:
            continue

        trial_number = int(match.group(1))
        location = match.group(2).strip()
        distance_m = int(match.group(3))
        track = normalize_track(location)

        meta_text = []
        node = subheader.parent.find_next_sibling("tr")
        while node is not None:
            text = normalize_space(node.get_text(" ", strip=True))
            if text.startswith("Horse"):
                break
            if text:
                meta_text.append(text)
            node = node.find_next_sibling("tr")
        going = ""
        sectional_time = ""
        batch_time = ""
        meta_blob = " ".join(meta_text)
        meta_match = re.search(r"Going:\s*(.*?)\s+Time:\s*([0-9:.]+)\s+Sectional Time:\s*(.*)", meta_blob, flags=re.IGNORECASE)
        if meta_match:
            going = normalize_space(meta_match.group(1))
            batch_time = normalize_space(meta_match.group(2))
            sectional_time = normalize_space(meta_match.group(3))

        table = anchor.find_next("table", class_=lambda c: c and "bigborder" in c)
        if table is None:
            continue
        tr_list = table.find_all("tr")
        if len(tr_list) < 2:
            continue
        headers = [normalize_space(td.get_text(" ", strip=True)).replace("\n", " ") for td in tr_list[0].find_all("td")]

        horse_number = 0
        for tr in tr_list[1:]:
            cells = tr.find_all("td")
            if not cells or not any(normalize_space(td.get_text(" ", strip=True)) for td in cells):
                continue
            horse_number += 1
            record: dict[str, object] = {
                "race_id": make_race_id(date_iso, trial_number),
                "Date": date_iso,
                "trial_number": trial_number,
                "horse_number": horse_number,
                "Track": track,
                "Dist.": distance_m,
                "going": going,
                "batch_time": batch_time,
                "sectional_time": sectional_time,
                "source_url": build_url(date_ddmmyyyy),
            }
            for header, cell in zip(headers, cells):
                text = normalize_space(cell.get_text(" ", strip=True))
                value = None if text == "--" else text
                record[header] = value
                anchor_tag = cell.find("a", href=True)
                if anchor_tag and header == "Horse":
                    href = anchor_tag["href"]
                    if href.startswith("/"):
                        href = f"https://racing.hkjc.com{href}"
                    record["horse_id"] = extract_horse_id(href)
            rows.append(record)

    columns = [
        "race_id", "Date", "trial_number", "horse_number", "Horse", "Jockey", "Trainer", "Draw",
        "Gear", "LBW", "Running Position", "Time", "Result", "Comment", "Track", "Dist.",
        "horse_id", "going", "batch_time", "sectional_time", "source_url",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


def scrape_date(date_ddmmyyyy: str) -> pd.DataFrame:
    return parse_trial_page(date_ddmmyyyy, fetch_html(build_url(date_ddmmyyyy)))


def run_year(year: int, workers: int, overwrite: bool) -> Path:
    dates = iter_year_dates(year)
    output_path = OUTPUT_DIR / f"barrier-trial-results-{year}.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")

    frames: list[pd.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(scrape_date, date): date for date in dates}
        done = 0
        total = len(future_map)
        for future in concurrent.futures.as_completed(future_map):
            date = future_map[future]
            done += 1
            try:
                df = future.result()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                log(f"[error] {date}: {exc}")
            if done % 25 == 0 or done == total:
                rows = sum(len(frame) for frame in frames)
                log(f"[{year}] processed {done}/{total} dates; rows={rows}")

    year_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not year_df.empty:
        year_df = year_df.sort_values(["Date", "trial_number", "horse_number"]).reset_index(drop=True)
    year_df.to_csv(output_path, index=False)
    log(f"[{year}] wrote {len(year_df):,} barrier trial rows -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Scrape HKJC barrier trials into yearly CSVs.")
    parser.add_argument("--years", nargs="+", type=int, default=[2025, 2026])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for year in sorted(set(args.years)):
        run_year(year, args.workers, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
