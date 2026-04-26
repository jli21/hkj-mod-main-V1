# HKJC Historical Data Coverage

This folder now contains a mix of direct HKJC scrapes and reconstructed historical datasets.

## Direct Scrapes

- `aspx-results/`
  - Yearly race result CSVs from `2000-2026`
- `barrier-trial-results/`
  - Yearly barrier trial CSVs from `2000-2026`
- `dividends/`
  - Yearly dividend CSVs from `2000-2026`
- `sectional-times/`
  - Sectional time CSVs already present in the repo
- `horse-profiles/horse-profiles.csv`
  - Horse profile backfill for all known `horse_id` values in the result history
  - Includes many retired horses when a profile page still resolves by `horseid`
- `race-incidents/`
  - Direct `racereportfull` scrape for `2000-2026`
- `comments-on-running/`
  - Direct `corunning` scrape backfilled for `2000-2026`
- `horse-movement-records/horse-movement-records.csv`
  - Direct `movementrecords` scrape for the recent `2025-2026` runner universe
- `horse-trackwork-records/`
  - Direct `trackworkresult` scrape for the recent `2025-2026` runner universe
  - Stored as yearly CSVs (`horse-trackwork-records-YYYY.csv`) to stay under GitHub file limits
- `horse-veterinary-records/horse-veterinary-records.csv`
  - Direct `ovehorse` scrape for the recent `2025-2026` runner universe

## Reconstructed Historical Datasets

These pages do not reliably return historical meeting content from HKJC for past race dates, so the yearly files below are derived from historical horse records and the direct recent horse-state tables above.

- `race-exceptional-factors/`
  - Yearly files from `2000-2026`
  - Derived from horse history, prior-run gaps, gear, trainer changes, and Conghua return timing where available
- `race-formline/`
  - Yearly files from `2000-2026`
  - Derived from shared prior races found in horse history
- `race-veterinary-records/`
  - Yearly files from `2000-2026`
  - Derived from horse-level veterinary history
  - Meaningful coverage starts in recent years only; older years may contain header-only files

## What Is Not Fully Available

- Direct historical `exceptionalfactors` HTML is not reliably available for old meetings
- Direct historical `formline` HTML is not reliably available for old meetings
- Direct historical race-day `veterinaryrecord` HTML is not reliably available for old meetings
- Older horse `movementrecords`, `trackworkresult`, and `ovehorse` pages are often empty or only useful for recent/active horses

## Practical Interpretation

- Use `race-incidents/` and `comments-on-running/` as direct textual evidence when present
- Use `race-exceptional-factors/`, `race-formline/`, and `race-veterinary-records/` as modeled historical proxies rather than literal archived HKJC page captures
- Treat blank yearly files in `race-veterinary-records/` and early `race-formline/` years as genuine coverage gaps, not scraper failures
