from __future__ import annotations

import concurrent.futures
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "historical-data"
RESULTS_DIR = DATA_DIR / "aspx-results"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)
PRINT_LOCK = threading.Lock()


def log(message: str) -> None:
    with PRINT_LOCK:
        print(message, flush=True)


def normalize_space(value: object | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


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


def absolute_url(url: str | None) -> str | None:
    if not url:
        return None
    if url.startswith("http://") or url.startswith("https://"):
        return url
    if url.startswith("/"):
        return f"https://racing.hkjc.com{url}"
    return url


def extract_query_value(url: str | None, key: str) -> str | None:
    if not url:
        return None
    try:
        query = parse_qs(urlparse(url).query)
        values = query.get(key)
        if values:
            return values[0]
    except Exception:
        pass
    match = re.search(fr"{re.escape(key)}=([A-Za-z0-9_]+)", url, flags=re.IGNORECASE)
    return match.group(1) if match else None


def parse_ddmmyyyy(value: str | None) -> str | None:
    text = normalize_space(value)
    if not text or text == "-":
        return None
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return text


def to_yyyymmdd(value: str) -> str:
    return value.replace("-", "")


def make_race_id(date_iso: str, race_number: int | str) -> str:
    return f"{to_yyyymmdd(date_iso)}{int(race_number):02d}"


def scrape_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_horse_ids(years: list[int] | None = None) -> list[str]:
    horse_ids: set[str] = set()
    for path in iter_result_files(years):
        df = pd.read_csv(path, usecols=["horse_id"], dtype=str)
        horse_ids.update(hid for hid in df["horse_id"].dropna().astype(str).tolist() if hid)
    return sorted(horse_ids)


def iter_result_files(years: list[int] | None = None) -> list[Path]:
    paths = sorted(RESULTS_DIR.glob("aspx-results-*.csv"))
    if years is None:
        return paths
    wanted = {int(year) for year in years}
    filtered: list[Path] = []
    for path in paths:
        try:
            year = int(path.stem.split("-")[-1])
        except ValueError:
            continue
        if year in wanted:
            filtered.append(path)
    return filtered


def load_runner_index(years: list[int] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["race_id", "Date", "race_number", "Horse No.", "Horse", "Track", "horse_id"]
    for path in iter_result_files(years):
        df = pd.read_csv(path, usecols=columns, dtype=str)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["race_id", "Date", "race_number", "Horse No.", "Horse", "Track", "horse_id"])
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["race_id", "Horse No.", "horse_id"])
    return out


def load_race_index(years: list[int] | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["race_id", "Date", "race_number", "Track"]
    for path in iter_result_files(years):
        df = pd.read_csv(path, usecols=columns, dtype=str).drop_duplicates()
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["race_id"])
    out["race_number"] = out["race_number"].astype(int)
    return out.sort_values(["Date", "race_number"]).reset_index(drop=True)


def meeting_index_from_races(races: pd.DataFrame) -> pd.DataFrame:
    meetings = races[["Date", "Track"]].drop_duplicates().rename(columns={"Track": "Racecourse"})
    return meetings.sort_values(["Date", "Racecourse"]).reset_index(drop=True)


def merge_runner_ids(df: pd.DataFrame, runner_index: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "horse_id" in df.columns and df["horse_id"].notna().any():
        return df
    merge_cols = [col for col in ["race_id", "Horse No."] if col in df.columns and col in runner_index.columns]
    if len(merge_cols) != 2:
        return df
    merged = df.merge(
        runner_index[["race_id", "Horse No.", "horse_id"]].drop_duplicates(),
        how="left",
        on=["race_id", "Horse No."],
        suffixes=("", "_runner"),
    )
    if "horse_id_runner" in merged.columns:
        merged["horse_id"] = merged["horse_id"].fillna(merged["horse_id_runner"])
        merged = merged.drop(columns=["horse_id_runner"])
    return merged


def run_parallel(items: list[object], worker_fn, workers: int, progress_label: str, progress_every: int = 100):
    results: list[object] = []
    failures: list[tuple[object, str]] = []
    total = len(items)
    if total == 0:
        return results, failures

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_map = {executor.submit(worker_fn, item): item for item in items}
        for done, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            item = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                failures.append((item, str(exc)))
            if done % progress_every == 0 or done == total:
                log(f"[{progress_label}] processed {done}/{total}; failures={len(failures)}")
    return results, failures
