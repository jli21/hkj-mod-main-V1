"""
prep_data.py
============
Data loading, merging, and preprocessing for HK horse racing.
Post-migration format: ISO dates, horse_id keys, flat dividend CSVs.
"""

import sys
import os
import pandas as pd
import numpy as np
import re
from typing import Any
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

_here = Path(__file__).resolve().parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(_here / ".." / "data-collection"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REFUND_CODES = {'WV', 'WV-A', 'WX', 'WX-A', 'WXNR'}
VOID_CODES   = {'PU', 'DNF', 'DISQ', 'UR', 'FE', 'TNP'}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_place(val: str) -> int:
    val = str(val).strip()
    if val in REFUND_CODES:
        return 100
    if val in VOID_CODES:
        return 99
    return int(val) if val.isnumeric() else pd.NA


def parse_time_to_seconds(t: Any) -> float:
    if pd.isna(t) or t is None:
        return np.nan
    t = str(t).strip()
    if t == '--':
        return np.nan
    try:
        if ':' in t:
            mins, rest = t.split(':', 1)
            secs, hund = rest.split('.', 1)
        else:
            mins, secs, hund = t.split('.')
        return int(mins) * 60 + int(secs) + int(hund) / 100
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Horse history merge (post-migration: keyed by horse_id, ISO dates)
# ---------------------------------------------------------------------------
def _load_horses(horses_dir: str) -> pd.DataFrame:
    """Load all horses_HK_*.json into a single DataFrame."""
    import json
    frames = []
    for f in sorted(Path(horses_dir).glob("horses_HK_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for horse_id, records in data.items():
            for rec in records:
                rec['horse_id'] = horse_id
            frames.extend(records)
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames)
    return df


def merge_historical_data(aspx_data: pd.DataFrame, horses_df: pd.DataFrame,
                          drop_unmatched: bool = False) -> pd.DataFrame:
    """
    Merge aspx race results with horse-specific records.
    Matches on horse_id + Date (a horse can only run once per day).
    """
    if horses_df.empty:
        print("Warning: horses_df is empty, returning aspx_data unchanged.")
        return aspx_data.copy()

    aspx = aspx_data.copy()
    n_before = len(aspx)

    # Deduplicate horse records: keep first per (horse_id, Date)
    horses_dedup = horses_df.drop_duplicates(subset=['horse_id', 'Date'], keep='first')

    merged = aspx.merge(
        horses_dedup,
        on=['horse_id', 'Date'],
        how='inner' if drop_unmatched else 'left',
        suffixes=('', '_horse'),
    )

    # Clean up duplicate columns from horse side
    drop_cols = [c for c in merged.columns if c.endswith('_horse')]
    merged.drop(columns=[c for c in drop_cols if c in merged.columns],
                inplace=True, errors='ignore')

    if drop_unmatched:
        n_dropped = n_before - len(merged)
        print(f"Dropped unmatched rows: {n_dropped} rows.")

    return merged


# ---------------------------------------------------------------------------
# Barrier trial feature extraction
# ---------------------------------------------------------------------------
def _load_barrier_trials(bt_dir: Path) -> pd.DataFrame:
    """Load and preprocess all barrier trial CSVs."""
    frames = []
    for f in sorted(bt_dir.glob('barrier-trial-results-*.csv')):
        df = pd.read_csv(f)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()

    bt = pd.concat(frames, ignore_index=True)
    bt['Date_dt'] = pd.to_datetime(bt['Date'], format='ISO8601')

    # Parse trial time (M.SS.HH format)
    def _parse_bt_time(t):
        try:
            parts = str(t).strip().split('.')
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 100
        except Exception:
            pass
        return np.nan

    bt['time_sec'] = bt['Time'].apply(_parse_bt_time)

    # Trial-level features
    bt['trial_key'] = bt['Date'].astype(str) + '_' + bt['trial_number'].astype(str)
    bt['winner_time'] = bt.groupby('trial_key')['time_sec'].transform('min')
    bt['bt_time_behind'] = bt['time_sec'] - bt['winner_time']
    bt['bt_field'] = bt.groupby('trial_key')['horse_number'].transform('count')

    # Early position
    def _parse_ep(rp):
        try:
            return int(str(rp).strip().split()[0])
        except Exception:
            return np.nan

    bt['early_pos'] = bt['Running Position'].apply(_parse_ep)
    bt['bt_early_pct'] = bt['early_pos'] / bt['bt_field']

    # Per-horse expanding stats
    bt = bt.sort_values('Date_dt')
    bt_horse = bt.groupby('horse_id')

    bt['_cum_early'] = (
        bt_horse['bt_early_pct'].expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    bt['_cum_behind'] = (
        bt_horse['bt_time_behind'].expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # Recent 3 trials behind
    tb1 = bt_horse['bt_time_behind'].shift(0)
    tb2 = bt_horse['bt_time_behind'].shift(1)
    tb3 = bt_horse['bt_time_behind'].shift(2)
    bt['_recent_behind'] = pd.concat([tb1, tb2, tb3], axis=1).mean(axis=1, skipna=True)

    # Speed figure: distance-normalized, z-scored within distance bucket
    bt['Dist'] = pd.to_numeric(bt.get('Dist.', bt.get('Dist', pd.Series())), errors='coerce')
    bt['_speed'] = bt['Dist'] / bt['time_sec']
    bt['_dist_bucket'] = pd.cut(bt['Dist'], bins=[0, 900, 1100, 1300, 1700, 9999],
                                 labels=['800', '1000', '1200', '1600', '1600+'])
    for db in bt['_dist_bucket'].cat.categories:
        mask = bt['_dist_bucket'] == db
        mean_spd = bt.loc[mask, '_speed'].mean()
        std_spd = bt.loc[mask, '_speed'].std()
        if std_spd > 0:
            bt.loc[mask, 'bt_speed_z'] = (bt.loc[mask, '_speed'] - mean_spd) / std_spd
    bt['bt_speed_z'] = bt['bt_speed_z'].fillna(0)

    # Expanding avg speed_z per horse
    bt['_cum_speed_z'] = (
        bt_horse['bt_speed_z'].expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    return bt


def _match_barrier_trials(df: pd.DataFrame, bt: pd.DataFrame) -> pd.DataFrame:
    """Match barrier trial features to race data (time-respecting)."""
    if bt.empty:
        df['bt_avg_early'] = 0.5
        df['bt_last_behind'] = 0.0
        df['bt_recent_behind'] = 0.0
        df['bt_speed_z'] = 0.0
        df['bt_last_speed_z'] = 0.0
        print("  WARNING: No barrier trial data found")
        return df

    # Build lookup: for each horse_id, list of (date, features)
    bt_lookup = {}
    for hid, grp in bt.groupby('horse_id'):
        bt_lookup[hid] = list(zip(
            grp['Date_dt'].values,
            grp['_cum_early'].values,
            grp['bt_time_behind'].values,
            grp['_recent_behind'].values,
            grp['_cum_speed_z'].values,
            grp['bt_speed_z'].values,
        ))

    df['bt_avg_early'] = np.nan
    df['bt_last_behind'] = np.nan
    df['bt_recent_behind'] = np.nan
    df['bt_speed_z'] = np.nan
    df['bt_last_speed_z'] = np.nan

    matched = 0
    for idx in df.index:
        hid = df.loc[idx, 'horse_id'] if 'horse_id' in df.columns else None
        if hid is None or hid not in bt_lookup:
            continue
        rd = df.loc[idx, 'r_date']
        for _date, _early, _last_bh, _rec_bh, _cum_spd, _last_spd in reversed(bt_lookup[hid]):
            if pd.Timestamp(_date) < rd:
                df.loc[idx, 'bt_avg_early'] = _early
                df.loc[idx, 'bt_last_behind'] = _last_bh
                df.loc[idx, 'bt_recent_behind'] = _rec_bh
                df.loc[idx, 'bt_speed_z'] = _cum_spd
                df.loc[idx, 'bt_last_speed_z'] = _last_spd
                matched += 1
                break

    df['bt_avg_early'] = df['bt_avg_early'].fillna(0.5)
    df['bt_last_behind'] = df['bt_last_behind'].fillna(df['bt_last_behind'].median())
    df['bt_recent_behind'] = df['bt_recent_behind'].fillna(df['bt_recent_behind'].median())
    df['bt_speed_z'] = df['bt_speed_z'].fillna(0)
    df['bt_last_speed_z'] = df['bt_last_speed_z'].fillna(0)

    print(f"  Barrier trials matched: {matched:,} / {len(df):,} ({matched/len(df):.1%})")
    print(f"  bt_avg_early: mean={df['bt_avg_early'].mean():.3f}")
    print(f"  bt_last_behind: mean={df['bt_last_behind'].mean():.3f}")
    print(f"  bt_recent_behind: mean={df['bt_recent_behind'].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# Sectional times feature extraction
# ---------------------------------------------------------------------------
def _load_sectional_times(st_dir: Path) -> pd.DataFrame:
    """Load sectional times, compute per-horse-per-race summary features."""
    print("Loading sectional times...")
    frames = []
    for f in sorted(st_dir.glob('sectional_times_*.csv')):
        df = pd.read_csv(f, low_memory=False,
                          usecols=['race_id', 'Date', 'horse_id', 'section_index',
                                   'running_position', 'sectional_time_sec',
                                   'race_sectional_time_sec', 'finish_place'])
        frames.append(df)
    if not frames:
        return pd.DataFrame()

    st = pd.concat(frames, ignore_index=True)
    st['Date'] = pd.to_datetime(st['Date'], errors='coerce')
    st['race_id'] = st['race_id'].astype(str)

    # Horse's sectional time minus race average at same section
    st['horse_vs_race'] = st['sectional_time_sec'] - st['race_sectional_time_sec']

    # Per-horse-per-race summary
    st_sorted = st.sort_values(['race_id', 'horse_id', 'section_index'])

    def _horse_race_summary(g):
        g = g.sort_values('section_index')
        n = len(g)
        if n < 2:
            return None

        vs_race = g['horse_vs_race'].values
        positions = g['running_position'].values

        # Closing speed: last section vs race (negative = closed fast)
        closing_speed = vs_race[-1] if len(vs_race) > 0 else 0

        # Early speed: first section vs race
        early_speed = vs_race[0] if len(vs_race) > 0 else 0

        # Best single-section time vs race (most negative = peak speed)
        best_vs_race = np.min(vs_race) if len(vs_race) > 0 else 0

        # When was the peak? 0=early, 1=late
        peak_at = np.argmin(vs_race) / max(n - 1, 1) if len(vs_race) > 0 else 0.5

        # Avg speed across all sections vs race
        avg_vs_race = np.mean(vs_race) if len(vs_race) > 0 else 0

        # Biggest position gain in one section
        pos_deltas = np.diff(positions) if len(positions) > 1 else np.array([0])
        best_gain = -np.min(pos_deltas) if len(pos_deltas) > 0 else 0

        return pd.Series({
            'sect_closing': closing_speed,
            'sect_early': early_speed,
            'sect_best': best_vs_race,
            'sect_peak_at': peak_at,
            'sect_avg': avg_vs_race,
            'sect_best_gain': best_gain,
            'race_date': g['Date'].iloc[0],
        })

    print(f"Computing per-horse-race sectional summaries ({len(st):,} rows)...")
    summary = st_sorted.groupby(['race_id', 'horse_id']).apply(
        _horse_race_summary, include_groups=False).dropna()
    summary = summary.reset_index()
    summary['race_date'] = pd.to_datetime(summary['race_date'])
    print(f"  Sectional summaries: {len(summary):,}")
    return summary


def _match_sectionals(df: pd.DataFrame, sect: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing sectional features per horse (time-respecting)."""
    if sect.empty:
        for c in ['trail_sect_closing', 'trail_sect_peak_at', 'trail_sect_avg',
                  'trail_sect_best_gain']:
            df[c] = 0
        print("  WARNING: No sectional data found")
        return df

    sect_sorted = sect.sort_values('race_date')

    # Build lookup by horse
    print("Building horse-level sectional lookup...")
    sect_lookup = {}
    for hid, grp in sect_sorted.groupby('horse_id'):
        entries = []
        # Running expanding averages
        cum_closing = []
        cum_peak = []
        cum_avg = []
        cum_gain = []
        for _, row in grp.iterrows():
            cum_closing.append(row['sect_closing'])
            cum_peak.append(row['sect_peak_at'])
            cum_avg.append(row['sect_avg'])
            cum_gain.append(row['sect_best_gain'])
            entries.append({
                'date': row['race_date'],
                'closing': np.mean(cum_closing),
                'peak_at': np.mean(cum_peak),
                'avg': np.mean(cum_avg),
                'best_gain': np.mean(cum_gain),
            })
        sect_lookup[hid] = entries

    df['trail_sect_closing'] = np.nan
    df['trail_sect_peak_at'] = np.nan
    df['trail_sect_avg'] = np.nan
    df['trail_sect_best_gain'] = np.nan

    matched = 0
    for idx in df.index:
        hid = df.loc[idx, 'horse_id'] if 'horse_id' in df.columns else None
        if hid is None or hid not in sect_lookup:
            continue
        rd = df.loc[idx, 'r_date']
        # Find most recent sectional strictly before race date
        for entry in reversed(sect_lookup[hid]):
            if pd.Timestamp(entry['date']) < rd:
                df.loc[idx, 'trail_sect_closing'] = entry['closing']
                df.loc[idx, 'trail_sect_peak_at'] = entry['peak_at']
                df.loc[idx, 'trail_sect_avg'] = entry['avg']
                df.loc[idx, 'trail_sect_best_gain'] = entry['best_gain']
                matched += 1
                break

    df['trail_sect_closing'] = df['trail_sect_closing'].fillna(0)
    df['trail_sect_peak_at'] = df['trail_sect_peak_at'].fillna(0.5)
    df['trail_sect_avg'] = df['trail_sect_avg'].fillna(0)
    df['trail_sect_best_gain'] = df['trail_sect_best_gain'].fillna(0)

    print(f"  Sectionals matched: {matched:,} / {len(df):,} ({matched/len(df):.1%})")
    print(f"  trail_sect_closing: mean={df['trail_sect_closing'].mean():.3f}")
    print(f"  trail_sect_peak_at: mean={df['trail_sect_peak_at'].mean():.3f}")
    print(f"  trail_sect_avg: mean={df['trail_sect_avg'].mean():.3f}")
    print(f"  trail_sect_best_gain: mean={df['trail_sect_best_gain'].mean():.3f}")
    return df


# ---------------------------------------------------------------------------
# Edge feature engineering (time-respecting)
# ---------------------------------------------------------------------------
def compute_edge_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute validated edge features with tested transformations.
    All time-respecting (shifted / expanding on past data only).
    """
    print("\nComputing edge features...")
    df = df.sort_values('r_date')

    horse = df.groupby('horse_id')
    field = df.groupby(level=0)
    df['field_size'] = field['y.status_place'].transform('count')

    # ------------------------------------------------------------------
    # 1. CLASS CHANGE + CLASS_DROP x ODDS_STABLE interaction
    # ------------------------------------------------------------------
    df['prev_class'] = horse['RaceClass_ord'].shift(1)
    df['class_change'] = df['prev_class'] - df['RaceClass_ord']
    df['class_change'] = df['class_change'].fillna(0)

    df['prev_odds'] = horse['Win Odds'].shift(1)
    _odds_ratio = np.log(df['Win Odds'] / df['prev_odds'].clip(lower=0.5))
    _odds_stable = _odds_ratio.abs() < 0.2
    df['class_drop_odds_stable'] = (
        (df['class_change'] >= 1) & _odds_stable & df['prev_odds'].notna()
    ).astype(int)

    print(f"  class_change: non-zero={(df['class_change'] != 0).sum():,}")
    print(f"  class_drop_odds_stable: hits={df['class_drop_odds_stable'].sum():,}")

    # ------------------------------------------------------------------
    # 2. CARRY WEIGHT CHANGE
    # ------------------------------------------------------------------
    df['prev_wt'] = horse['Act_Wt'].shift(1)
    df['weight_change'] = df['Act_Wt'] - df['prev_wt']
    df['weight_change'] = df['weight_change'].fillna(0)
    print(f"  weight_change: non-zero={(df['weight_change'] != 0).sum():,}")

    # ------------------------------------------------------------------
    # 3. CARRY WEIGHT Z-SCORE (within race)
    # ------------------------------------------------------------------
    race_wt_mean = field['Act_Wt'].transform('mean')
    race_wt_std = field['Act_Wt'].transform('std').replace(0, np.nan)
    df['wt_z'] = (df['Act_Wt'] - race_wt_mean) / race_wt_std
    df['wt_z'] = df['wt_z'].fillna(0)
    print(f"  wt_z: mean={df['wt_z'].mean():.3f}, std={df['wt_z'].std():.3f}")

    # ------------------------------------------------------------------
    # 4. ODDS CHANGE
    # ------------------------------------------------------------------
    df['odds_change'] = df['Win Odds'] - df['prev_odds']
    df['odds_change'] = df['odds_change'].fillna(0)
    print(f"  odds_change: non-zero={(df['odds_change'] != 0).sum():,}")

    # ------------------------------------------------------------------
    # 5. RECENT FORM — avg 1/place last 3 (better encoding: amplifies 1st-4th, compresses back)
    # ------------------------------------------------------------------
    p1 = horse['y.status_place'].shift(1)
    p2 = horse['y.status_place'].shift(2)
    p3 = horse['y.status_place'].shift(3)
    ip1 = 1.0 / p1.clip(lower=1)
    ip2 = 1.0 / p2.clip(lower=1)
    ip3 = 1.0 / p3.clip(lower=1)
    df['recent_form'] = pd.concat([ip1, ip2, ip3], axis=1).mean(axis=1, skipna=True)
    df['recent_form'] = df['recent_form'].fillna(1.0 / 7.0)  # default ~7th place equivalent
    print(f"  recent_form (1/place): mean={df['recent_form'].mean():.4f}, std={df['recent_form'].std():.4f}")

    # FORM VS CAREER — is horse above/below its own career average? (r=-0.25, low odds corr)
    df['_cum_inv_place'] = (
        horse['y.status_place'].expanding(min_periods=1)
        .apply(lambda x: (1.0 / x.clip(lower=1)).mean(), raw=False)
        .reset_index(level=0, drop=True)
    )
    df['career_form'] = horse['_cum_inv_place'].shift(1)
    df['form_vs_career'] = df['recent_form'] - df['career_form']  # positive = recent better than career
    df['form_vs_career'] = df['form_vs_career'].fillna(0)
    df.drop(columns=['_cum_inv_place', 'career_form'], inplace=True)
    print(f"  form_vs_career: mean={df['form_vs_career'].mean():.4f}, std={df['form_vs_career'].std():.4f}")

    # ------------------------------------------------------------------
    # 6. RUNNING STYLE (trailing avg of early_pos / field_size)
    # ------------------------------------------------------------------
    if 'early_pos' in df.columns:
        field_size = field['early_pos'].transform('count')
        df['early_pct'] = df['early_pos'] / field_size

        df['_cum_style'] = (
            df.groupby('horse_id')['early_pct']
            .expanding(min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['running_style'] = horse['_cum_style'].shift(1)
        df['running_style'] = df['running_style'].fillna(0.5)
        df.drop(columns=['_cum_style', 'early_pct'], inplace=True)
        print(f"  running_style: mean={df['running_style'].mean():.3f}, std={df['running_style'].std():.3f}")
    else:
        df['running_style'] = 0.5
        print("  WARNING: early_pos not available, running_style set to 0.5")

    # ------------------------------------------------------------------
    # 6b. TRAILING ACCELERATION
    # ------------------------------------------------------------------
    if 'race_acceleration' in df.columns:
        df['_cum_accel'] = (
            df.groupby('horse_id')['race_acceleration']
            .expanding(min_periods=2)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df['trail_acceleration'] = horse['_cum_accel'].shift(1)
        df['trail_acceleration'] = df['trail_acceleration'].fillna(0)
        df.drop(columns=['_cum_accel'], inplace=True)
        print(f"  trail_acceleration: mean={df['trail_acceleration'].mean():.4f}, "
              f"std={df['trail_acceleration'].std():.4f}")
    else:
        df['trail_acceleration'] = 0
        print("  WARNING: race_acceleration not available")

    # ------------------------------------------------------------------
    # 7. TRAINER-TRACK SPECIALIZATION
    # ------------------------------------------------------------------
    if 'Trainer' in df.columns:
        df['_t_win'] = (df['y.status_place'] == 1).astype(int)

        df['_t_cum_wins'] = df.groupby('Trainer')['_t_win'].cumsum() - df['_t_win']
        df['_t_cum_runs'] = df.groupby('Trainer').cumcount()
        df['_t_wr'] = df['_t_cum_wins'] / df['_t_cum_runs'].replace(0, np.nan)

        df['_t_track'] = df['Trainer'].astype(str) + '|' + df['Racecourse'].astype(str)
        df['_tt_cum_wins'] = df.groupby('_t_track')['_t_win'].cumsum() - df['_t_win']
        df['_tt_cum_runs'] = df.groupby('_t_track').cumcount()
        df['_tt_wr'] = df['_tt_cum_wins'] / df['_tt_cum_runs'].replace(0, np.nan)

        df['trainer_track_spec'] = np.where(
            (df['_tt_cum_runs'] >= 20) & (df['_t_cum_runs'] >= 50),
            df['_tt_wr'] - df['_t_wr'],
            0.0
        )

        # MARKET MISPRICING: trainer×track actual WR vs market-implied WR
        # Positive = trainer outperforms market expectation at this track
        df['_tt_impl_cum'] = df.groupby('_t_track')['Implied_Prob'].cumsum() - df['Implied_Prob']
        df['_tt_mkt_wr'] = df['_tt_impl_cum'] / df['_tt_cum_runs'].replace(0, np.nan)
        df['tt_gap'] = np.where(
            df['_tt_cum_runs'] >= 20,
            df['_tt_wr'] - df['_tt_mkt_wr'],
            0.0
        )

        # MARKET MISPRICING: trainer overall actual WR vs market-implied WR
        df['_t_impl_cum'] = df.groupby('Trainer')['Implied_Prob'].cumsum() - df['Implied_Prob']
        df['_t_mkt_wr'] = df['_t_impl_cum'] / df['_t_cum_runs'].replace(0, np.nan)
        df['tr_gap'] = np.where(
            df['_t_cum_runs'] >= 30,
            df['_t_wr'] - df['_t_mkt_wr'],
            0.0
        )

        print(f"  tt_gap: non-zero={(df['tt_gap'] != 0).sum():,}, mean={df['tt_gap'].mean():.5f}")
        print(f"  tr_gap: non-zero={(df['tr_gap'] != 0).sum():,}, mean={df['tr_gap'].mean():.5f}")

        drop_tmp = [c for c in df.columns if c.startswith('_t')]
        df.drop(columns=drop_tmp, inplace=True)
        print(f"  trainer_track_spec: non-zero={(df['trainer_track_spec'] != 0).sum():,}")
    else:
        df['trainer_track_spec'] = 0.0

    # ------------------------------------------------------------------
    # 8. GOING CHANGE
    # ------------------------------------------------------------------
    df['prev_going'] = horse['Going_ORD'].shift(1)
    df['going_change'] = df['Going_ORD'] - df['prev_going']
    df['going_change'] = df['going_change'].fillna(0)
    print(f"  going_change: non-zero={(df['going_change'] != 0).sum():,}")

    # ------------------------------------------------------------------
    # 9. DRAW x RACECOURSE
    # ------------------------------------------------------------------
    df['draw_outside_ST'] = ((df['Draw'] >= 10) & (df['Racecourse'] == 1)).astype(int)
    df['draw_inside_HV'] = ((df['Draw'] <= 3) & (df['Racecourse'] == 0)).astype(int)
    print(f"  draw_outside_ST: {df['draw_outside_ST'].sum():,}, "
          f"draw_inside_HV: {df['draw_inside_HV'].sum():,}")

    # ------------------------------------------------------------------
    # 10. FIELD FORM DISPERSION
    # ------------------------------------------------------------------
    field_form_mean = field['recent_form'].transform('mean').clip(1)
    field_form_std = field['recent_form'].transform('std')
    df['field_form_cv'] = field_form_std / field_form_mean
    print(f"  field_form_cv: mean={df['field_form_cv'].mean():.3f}, std={df['field_form_cv'].std():.3f}")

    # ------------------------------------------------------------------
    # 11. BEHAVIORAL BIAS FEATURES
    # ------------------------------------------------------------------
    # TOP-3 COUNT in last 5 races (continuous, replaces binary streak_good)
    # +3.4% spread at 1-3 (14/15 yrs), +0.7% at 6-10 (9/15 yrs)
    _p1 = horse['y.status_place'].shift(1)
    _p2 = horse['y.status_place'].shift(2)
    _p3 = horse['y.status_place'].shift(3)
    _p4 = horse['y.status_place'].shift(4)
    _p5 = horse['y.status_place'].shift(5)
    _t1 = (_p1 <= 3).astype(float)
    _t2 = (_p2 <= 3).astype(float)
    _t3 = (_p3 <= 3).astype(float)
    _t4 = (_p4 <= 3).astype(float)
    _t5 = (_p5 <= 3).astype(float)
    df['top3_count_5'] = pd.concat([_t1, _t2, _t3, _t4, _t5], axis=1).sum(axis=1, skipna=True)
    df['top3_count_5'] = df['top3_count_5'].fillna(0)

    df['prev_win'] = (_p1 == 1).astype(int).fillna(0)

    max_impl = field['Implied_Prob'].transform('max')
    df['is_fav'] = (df['Implied_Prob'] == max_impl).astype(int)
    df['fav_field_size'] = df['is_fav'] * df['field_size']

    df['prev_date'] = horse['r_date'].shift(1)
    df['days_since'] = (df['r_date'] - df['prev_date']).dt.days.fillna(30)

    # SETUP WEIGHT Z — is horse lighter than its own history?
    # +4.6% spread at 1-3 (12/15), +1.1% at 3-6 (12/15), +0.9% at 10-20 (9/15)
    # Replaces trainer_moves (crude count)
    df['_horse_avg_wt'] = (
        horse['Act_Wt'].expanding(min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )
    df['_horse_avg_wt_shifted'] = horse['_horse_avg_wt'].shift(1)
    df['_horse_std_wt'] = (
        horse['Act_Wt'].expanding(min_periods=2).std()
        .reset_index(level=0, drop=True)
    )
    df['_horse_std_wt_shifted'] = horse['_horse_std_wt'].shift(1).fillna(3)
    df['setup_weight_z'] = (
        (df['_horse_avg_wt_shifted'] - df['Act_Wt']) /
        df['_horse_std_wt_shifted'].clip(lower=1)
    )
    df['setup_weight_z'] = df['setup_weight_z'].fillna(0)
    df.drop(columns=[c for c in df.columns if c.startswith('_horse_')], inplace=True)

    # CAREER BEAT ODDS
    df['_beat_odds'] = (df['y.status_place'] == 1).astype(float) - df['Implied_Prob']
    df['_cum_beat'] = (
        horse['_beat_odds'].expanding(min_periods=3).mean()
        .reset_index(level=0, drop=True)
    )
    df['career_beat_odds'] = horse['_cum_beat'].shift(1)
    df['career_beat_odds'] = df['career_beat_odds'].fillna(0)

    print(f"  prev_win: {df['prev_win'].sum():,}")
    print(f"  top3_count_5: mean={df['top3_count_5'].mean():.2f}")
    print(f"  fav_field_size: favs={df['is_fav'].sum():,}")
    print(f"  setup_weight_z: mean={df['setup_weight_z'].mean():.4f}, std={df['setup_weight_z'].std():.4f}")
    print(f"  career_beat_odds: non-zero={(df['career_beat_odds'] != 0).sum():,}, "
          f"mean={df['career_beat_odds'].mean():.5f}")
    print(f"  days_since: mean={df['days_since'].mean():.1f}")

    # ------------------------------------------------------------------
    # 12. BARRIER TRIAL FEATURES
    # ------------------------------------------------------------------
    bt_dir = _here / '..' / 'data' / 'historical-data' / 'barrier-trial-results'
    if bt_dir.exists():
        print("\n  Loading barrier trial data...")
        bt = _load_barrier_trials(bt_dir)
        df = _match_barrier_trials(df, bt)
    else:
        df['bt_avg_early'] = 0.5
        df['bt_last_behind'] = 0.0
        df['bt_recent_behind'] = 0.0
        print("  WARNING: No barrier trial data found")

    # ------------------------------------------------------------------
    # 13. SECTIONAL TIMES FEATURES (trailing per horse)
    # ------------------------------------------------------------------
    st_dir = _here / '..' / 'data' / 'historical-data' / 'sectional-times'
    if st_dir.exists():
        print("\n  Loading sectional times data...")
        sect = _load_sectional_times(st_dir)
        df = _match_sectionals(df, sect)
    else:
        df['trail_sect_closing'] = 0
        df['trail_sect_peak_at'] = 0.5
        df['trail_sect_avg'] = 0
        df['trail_sect_best_gain'] = 0
        print("  WARNING: No sectional times data found")

    # ------------------------------------------------------------------
    # 14. BODY WEIGHT FEATURES (from Declar. Horse Wt.)
    #     Builds body_wt_change + market-failure residual bucketed by
    #     (body_wt_change × Implied_Prob). Residual is the production feature.
    # ------------------------------------------------------------------
    if 'Declar. Horse Wt.' in df.columns and df['Declar. Horse Wt.'].notna().any():
        print("\n  Computing body weight features...")
        df['body_wt_raw'] = df['Declar. Horse Wt.']
        # Per-horse change (time-respecting via shift)
        df['body_wt_change'] = (
            df['body_wt_raw'] - horse['body_wt_raw'].shift(1)
        ).fillna(0)

        # Market-failure residual, bucketed by (body_wt_change × Implied_Prob).
        # Time-respecting expanding: (past_wins − past_implied) / (past_implied + k)
        wc_bins = [-10000, -10, -5, 5, 10, 10000]
        wc_labels = ['wc_hd', 'wc_md', 'wc_flat', 'wc_mg', 'wc_hg']
        impl_bins = [0, 0.04, 0.08, 0.15, 0.25, 1.01]
        impl_labels = ['i_long', 'i_medl', 'i_mid', 'i_fav', 'i_vfav']

        _wc = pd.cut(df['body_wt_change'], bins=wc_bins, labels=wc_labels)
        _impl = pd.cut(df['Implied_Prob'], bins=impl_bins, labels=impl_labels)
        df['_bwt_bucket'] = _wc.astype(str) + '|' + _impl.astype(str)
        df['_bwt_win'] = (df['y.status_place'] == 1).astype(float)

        # Sort by date for time-respecting expanding sums
        df_sorted = df.sort_values('r_date')
        stat_rows = df_sorted.loc[df_sorted['y.status_place'] != 99].copy()
        grp = stat_rows.groupby('_bwt_bucket', sort=False)
        cum_wins_past = grp['_bwt_win'].cumsum() - stat_rows['_bwt_win']
        cum_impl_past = grp['Implied_Prob'].cumsum() - stat_rows['Implied_Prob']
        k = 50.0
        residual = (cum_wins_past - cum_impl_past) / (cum_impl_past + k)

        df['body_wt_mkt_residual'] = 0.0
        df.loc[residual.index, 'body_wt_mkt_residual'] = residual.values
        df['body_wt_mkt_residual'] = df['body_wt_mkt_residual'].fillna(0)

        df.drop(columns=['_bwt_bucket', '_bwt_win'], inplace=True)
        print(f"  body_wt_mkt_residual: mean={df['body_wt_mkt_residual'].mean():+.5f}, "
              f"std={df['body_wt_mkt_residual'].std():.4f}, "
              f"corr(impl)={df['body_wt_mkt_residual'].corr(df['Implied_Prob']):+.4f}")
    else:
        df['body_wt_raw'] = np.nan
        df['body_wt_change'] = 0
        df['body_wt_mkt_residual'] = 0
        print("  WARNING: No Declar. Horse Wt. column available")

    # ------------------------------------------------------------------
    # Clean up
    # ------------------------------------------------------------------
    for c in ['prev_class', 'prev_wt', 'prev_odds', 'prev_going', 'prev_date',
              'prev_jockey', 'is_fav', '_beat_odds', '_cum_beat']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    df.sort_index(inplace=True)

    # Current production feature set (prune2 + body_wt_mkt_residual)
    # PRUNED in prior refactor: weight_change, days_since, draw_inside_HV,
    #   class_drop_odds_stable, bt_recent_behind, top3_count_5, field_form_cv.
    # top3_count_5 and field_form_cv computed above but excluded from edge_cols.
    edge_cols = ['class_change',
                 'wt_z', 'recent_form',
                 'trainer_track_spec',
                 'draw_outside_ST',
                 'prev_win', 'fav_field_size',
                 'form_vs_career',
                 'setup_weight_z', 'career_beat_odds',
                 'bt_avg_early', 'bt_last_behind',
                 'trail_sect_closing', 'trail_sect_peak_at',
                 'trail_sect_avg', 'trail_sect_best_gain',
                 'body_wt_mkt_residual']
    print(f"\nEdge features computed ({len(edge_cols)}): {', '.join(edge_cols)}")
    return df


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------
def prep_data(df_raw: pd.DataFrame, drop_void: bool = False) -> pd.DataFrame:

    def _drop_void_races(df: pd.DataFrame) -> pd.DataFrame:
        void_races = df.loc[df['Place'] == 99, 'race_id'].unique()
        if len(void_races):
            print(f"Removing {len(void_races)} races with void placings.")
        return df[~df['race_id'].isin(void_races)]

    print(
        "Place code mapping:\n"
        "  WV/WV-A/WX/WX-A/WXNR -> 100 (refunded)\n"
        "  PU/DNF/DISQ/UR/FE/TNP -> 99  (void)\n"
    )

    df = df_raw.copy()

    # race_id already exists from migration (YYYYMMDDRR)
    # Ensure it's a string
    df['race_id'] = df['race_id'].astype(str)

    df = df[df['Horse No.'].notna()]
    df['Horse No.'] = df['Horse No.'].astype(int)
    df['Place'] = df['Pla.'].apply(_normalise_place).astype('Int64')

    if drop_void:
        df = _drop_void_races(df)

    print("Dropping refunded runners (Place = 100).")
    df = df[df['Place'] != 100]

    # ------------------------------------------------------------------
    # Numeric fields
    # ------------------------------------------------------------------
    df['Win Odds'] = pd.to_numeric(df['Win Odds'], errors='raise')
    df['Implied_Prob'] = 1 / df['Win Odds']
    df['Implied_Prob'] /= df.groupby('race_id')['Implied_Prob'].transform('sum')

    df['log_implied_prob'] = np.log(df['Implied_Prob'].clip(lower=1e-8))
    df['market_rank'] = df.groupby('race_id')['Implied_Prob'].rank(ascending=False, method='min')
    df['market_rank'] = df['market_rank'].fillna(df['market_rank'].max()).astype(int)

    # RC/Track/Course parsing from horse history (if present)
    if 'RC/Track/ Course' in df.columns:
        df[['RC', 'Track', 'Course']] = (
            df['RC/Track/ Course'].str.replace('"', '').str.strip()
              .str.split(' / ', expand=True)
        )
        df['RC'] = df['RC'].astype('category')
    else:
        # Use Track column from aspx-results directly
        df['RC'] = df['Track'].astype('category') if 'Track' in df.columns else 'ST'

    df['Rtg.']  = pd.to_numeric(df.get('Rtg.'),  errors='coerce') if 'Rtg.' in df.columns else np.nan
    df['Draw']  = pd.to_numeric(df['Dr.'],    errors='coerce') if 'Dr.' in df.columns else np.nan
    if 'Dr.' in df.columns:
        df['Dr.'] = df['Dr.'].astype('category')
    df['Dist']  = pd.to_numeric(df.get('Dist.'),  errors='coerce') if 'Dist.' in df.columns else np.nan
    if 'Dist.' in df.columns:
        df['Dist.'] = df['Dist.'].astype('category')
    df['Declar. Horse Wt.'] = pd.to_numeric(df.get('Declar. Horse Wt.'), errors='coerce') if 'Declar. Horse Wt.' in df.columns else np.nan
    df['Act_Wt']            = pd.to_numeric(df['Act. Wt.'],          errors='coerce')
    df['TimeSec'] = df['Finish Time'].apply(parse_time_to_seconds)
    if 'Course' in df.columns:
        df['Course'] = df['Course'].astype('category')

    # ------------------------------------------------------------------
    # Rating normalisation
    # ------------------------------------------------------------------
    if 'Rtg.' in df.columns and df['Rtg.'].notna().any():
        grp_mean = df.groupby('race_id')['Rtg.'].transform('mean')
        grp_std  = df.groupby('race_id')['Rtg.'].transform('std')
        df['Rtg_norm'] = (df['Rtg.'] - grp_mean) / grp_std.replace(0, np.nan)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------
    df['y.status_place'] = df['Place'].astype(int)
    df['y.status_win']   = (df['Place'] == 1).astype(int)
    df['Returns'] = np.where(
        df['Place'] == 100, 1.0,
        np.where(df['y.status_win'] == 1, df['Win Odds'], 0.0),
    )

    # ------------------------------------------------------------------
    # Dates & index
    # ------------------------------------------------------------------
    df['Date'] = pd.to_datetime(df['Date'], format='ISO8601')
    df['Year'] = df['Date'].dt.year
    df.set_index(['race_id', 'Horse No.'], inplace=True)
    df.sort_index(inplace=True)

    # Deduplicate
    n_before = len(df)
    df = df[~df.index.duplicated(keep='first')].copy()
    if n_before - len(df) > 0:
        print(f"Dropped {n_before - len(df)} duplicate (race_id, Horse No.) entries.")

    # ------------------------------------------------------------------
    # Class 1-5 only
    # ------------------------------------------------------------------
    if 'Race Class' in df.columns:
        df = df[df['Race Class'].isin(['1', '2', '3', '4', '5'])].copy()
        df['RaceClass_ord'] = df['Race Class'].astype(int)
        df['Race Class'] = df['Race Class'].astype('category')

    # ------------------------------------------------------------------
    # Racecourse encoding
    # ------------------------------------------------------------------
    rc_map = {'ST': 1, 'HV': 0}
    df = df[df['RC'].isin(rc_map)].copy()
    df['Racecourse'] = df['RC'].map(rc_map).astype(int)

    # ------------------------------------------------------------------
    # Track encoding
    # ------------------------------------------------------------------
    if 'Track' in df.columns:
        track_map = {'Turf': 1, 'AWT': 0}
        df['Track_Turf'] = df['Track'].map(track_map).fillna(1).astype(int)
        df['Track'] = df['Track'].astype('category')

    # ------------------------------------------------------------------
    # Going encoding (ordinal)
    # ------------------------------------------------------------------
    going_order = [
        'HEAVY', 'WET SLOW', 'SLOW',
        'YIELDING TO SOFT', 'YIELDING',
        'SOFT', 'GOOD TO YIELDING',
        'GOOD', 'GOOD TO FIRM',
        'FAST', 'WET FAST',
    ]
    df['Going'] = df['Going'].str.upper().str.strip()
    cat_type = pd.CategoricalDtype(categories=going_order, ordered=True)
    df['Going'] = df['Going'].astype(cat_type)
    df['Going_ORD'] = df['Going'].cat.codes.astype('Int64')

    # ------------------------------------------------------------------
    # Log transforms
    # ------------------------------------------------------------------
    df['log_time'] = np.log(df['TimeSec'])
    if 'Dist' in df.columns:
        df['log_dist'] = np.log(df['Dist'])

    # ------------------------------------------------------------------
    # Aliases
    # ------------------------------------------------------------------
    df['time_sec'] = df['TimeSec']
    df['r_date']   = df['Date']
    df['Rider']    = df['Jockey']

    # ------------------------------------------------------------------
    # Parse Running Position -> early_pos + acceleration
    # ------------------------------------------------------------------
    if 'Running Position' in df.columns:
        def _parse_rp(rp):
            try:
                parts = [int(p) for p in str(rp).strip().split()]
                if len(parts) >= 2:
                    return parts
            except Exception:
                pass
            return []

        rp_parsed = df['Running Position'].apply(_parse_rp)
        df['early_pos'] = rp_parsed.apply(lambda p: p[0] if len(p) >= 1 else np.nan)

        field_size = df.groupby(level=0)['early_pos'].transform('count')
        def _accel(parts, fs):
            if len(parts) >= 2 and fs > 1:
                norm = [(p - 1) / max(fs - 1, 1) for p in parts]
                return (norm[-1] - norm[0]) / max(len(parts) - 1, 1)
            return np.nan
        df['race_acceleration'] = [_accel(rp, fs) for rp, fs in zip(rp_parsed, field_size)]

    return df


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    script_dir = str(_here)
    project_root = os.path.dirname(script_dir)
    data_dir = os.path.join(project_root, "data")
    historical_dir = os.path.join(data_dir, "historical-data")
    processed_dir = os.path.join(data_dir, "processed")

    years = range(2010, 2027)

    # Load horse history (post-migration: keyed by horse_id, ISO dates)
    horses_dir = os.path.join(historical_dir, "horses")
    print("Loading horse history...")
    horses_df = _load_horses(horses_dir)
    print(f"Loaded {len(horses_df):,} horse records")

    # Load and merge aspx results
    df_all = pd.DataFrame()
    for year in years:
        print(f'Year = {year}... Joining dataframe....')
        filename = os.path.join(historical_dir, "aspx-results", f"aspx-results-{year}.csv")
        df = pd.read_csv(filename)
        merged_df = merge_historical_data(df, horses_df, drop_unmatched=True)
        df_all = pd.concat([df_all, merged_df], ignore_index=True)

    print("\n" + "=" * 80)
    print("PREPROCESSING DATA")
    print("=" * 80)
    base = prep_data(df_all)

    # ------------------------------------------------------------------
    # Edge features
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("COMPUTING EDGE FEATURES")
    print("=" * 80)
    base = compute_edge_features(base)

    # Current production feature set (prune2 + body_wt_mkt_residual = 19 edge)
    edge_feature_cols = [
        'class_change',
        'wt_z', 'recent_form',
        'trainer_track_spec', 'tt_gap', 'tr_gap',
        'draw_outside_ST',
        'prev_win', 'fav_field_size',
        'form_vs_career',
        'setup_weight_z', 'career_beat_odds',
        'bt_avg_early', 'bt_last_behind',
        'trail_sect_closing', 'trail_sect_peak_at',
        'trail_sect_avg', 'trail_sect_best_gain',
        'body_wt_mkt_residual',
    ]

    # Save base features
    feature_cols = [
        'Implied_Prob', 'log_implied_prob',
        'Racecourse', 'Track_Turf',
        'Draw', 'Act_Wt',
        'Year', 'Win Odds', 'y.status_place',
    ] + edge_feature_cols
    os.makedirs(processed_dir, exist_ok=True)
    print(f"\nSaving base features to race_features.parquet...")
    base[feature_cols].to_parquet(os.path.join(processed_dir, "race_features.parquet"))
    print(f"Saved {len(base):,} rows with {len(feature_cols)} feature columns")

    # Save extended features
    extended_cols = feature_cols + [
        'log_time', 'log_dist', 'RaceClass_ord', 'Racecourse', 'Track_Turf',
        'Going_ORD', 'Act_Wt', 'r_date', 'horse_id', 'Rider', 'time_sec',
        'TimeSec', 'Date', 'Jockey', 'Trainer',
    ]
    extended_cols = [c for c in dict.fromkeys(extended_cols) if c in base.columns]
    print(f"\nSaving extended features to race_features_extended.parquet...")
    base[extended_cols].to_parquet(os.path.join(processed_dir, "race_features_extended.parquet"))
    print(f"Saved {len(base):,} rows with {len(extended_cols)} extended feature columns")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETE")
    print("=" * 80)
