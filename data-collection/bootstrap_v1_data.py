from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = ROOT.parent / "hkjc-mod" / "data" / "historical-data"
TARGET_DATA = ROOT / "data" / "historical-data"

sys.path.insert(0, str(ROOT))
from migrate_data import migrate_aspx, migrate_barrier_trials, migrate_dividends, migrate_horses  # noqa: E402


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path, overwrite: bool) -> bool:
    if not src.exists():
        return False
    if dst.exists() and not overwrite:
        return True
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return True


def bootstrap_files(source_root: Path, years: list[int], overwrite: bool) -> dict[str, int]:
    counts = {
        "horses": 0,
        "aspx": 0,
        "dividends": 0,
        "barrier": 0,
    }

    horse_src = source_root / "horses"
    horse_dst = TARGET_DATA / "horses"
    ensure_dir(horse_dst)
    for src in sorted(horse_src.glob("horses_HK_*.json")):
        dst = horse_dst / src.name
        if copy_file(src, dst, overwrite):
            counts["horses"] += 1

    for year in years:
        aspx_src = source_root / "aspx-results" / f"aspx-results-{year}.csv"
        aspx_dst = TARGET_DATA / "aspx-results" / aspx_src.name
        if copy_file(aspx_src, aspx_dst, overwrite):
            counts["aspx"] += 1

        div_src = source_root / "dividends" / f"dividends-{year}.json"
        div_dst = TARGET_DATA / "dividends" / div_src.name
        if copy_file(div_src, div_dst, overwrite):
            counts["dividends"] += 1

        bt_src = source_root / "barrier-trial-results" / f"barrier-trial-results-{year}.csv"
        bt_dst = TARGET_DATA / "barrier-trial-results" / bt_src.name
        if copy_file(bt_src, bt_dst, overwrite):
            counts["barrier"] += 1

    return counts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap V1 historical data from sibling repo and migrate it into V1 format."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Sibling historical-data directory",
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2025, 2026],
        help="Year files to copy for results, dividends, and barrier trials",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files when they already exist",
    )
    args = parser.parse_args()

    source_root = args.source_root
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    log(f"Bootstrapping from {source_root}")
    counts = bootstrap_files(source_root, sorted(set(args.years)), args.overwrite)
    log(
        "Copied files -> "
        f"horses={counts['horses']}, aspx={counts['aspx']}, "
        f"dividends={counts['dividends']}, barrier={counts['barrier']}"
    )

    log("Migrating ASPX results into V1 format...")
    migrate_aspx(dry_run=False)
    log("Migrating barrier trial files into V1 format...")
    migrate_barrier_trials(dry_run=False)
    log("Migrating dividend files into flat CSV format...")
    migrate_dividends(dry_run=False)
    log("Migrating horse history JSON into horse_id-keyed format...")
    migrate_horses(dry_run=False)
    log("Bootstrap and migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
