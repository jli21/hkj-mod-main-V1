"""Creative feature exploration — things beyond simple lag features."""
import numpy as np
import pandas as pd

blups = pd.read_parquet("data/processed/race_features_with_blups.parquet")
ext = pd.read_parquet("data/processed/race_features_extended.parquet")

# First: what columns do we actually have?
print("=" * 80)
print("ALL AVAILABLE COLUMNS")
print("=" * 80)
print("\nBLUP parquet columns:")
for c in sorted(blups.columns):
    print(f"  {c}: {blups[c].dtype}, nunique={blups[c].nunique()}, null={blups[c].isna().sum()}")

print("\nExtended parquet columns:")
for c in sorted(ext.columns):
    if c not in blups.columns:
        print(f"  {c}: {ext[c].dtype}, nunique={ext[c].nunique()}, null={ext[c].isna().sum()}")

# Let's also look at raw data to see if there's more
import os
raw_files = os.listdir("data/historical-data/aspx-results/")
print(f"\nRaw data files: {len(raw_files)}")
sample_raw = pd.read_csv(f"data/historical-data/aspx-results/{raw_files[0]}")
print(f"\nRaw CSV columns ({raw_files[0]}):")
for c in sorted(sample_raw.columns):
    print(f"  {c}: {sample_raw[c].dtype}, sample={sample_raw[c].dropna().head(3).tolist()}")

# Also check horses data
import json
horse_files = os.listdir("data/historical-data/horses/")
with open(f"data/historical-data/horses/{horse_files[0]}") as f:
    horses_sample = json.load(f)
    first_key = list(horses_sample.keys())[0]
    print(f"\nHorse data keys for '{first_key}':")
    horse_data = horses_sample[first_key]
    if isinstance(horse_data, dict):
        for k, v in horse_data.items():
            print(f"  {k}: {type(v).__name__}, sample={str(v)[:100]}")
    elif isinstance(horse_data, list) and len(horse_data) > 0:
        print(f"  List of {len(horse_data)} items")
        if isinstance(horse_data[0], dict):
            for k, v in horse_data[0].items():
                print(f"    {k}: {type(v).__name__}, sample={str(v)[:80]}")
