"""
ui_helpers.py
=============
Shared UI helper functions, data loading, constants, and model prediction
utilities for the Streamlit multipage dashboard.
"""

import sys
import os
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../data-collection")))
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../model")))

BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
HISTORICAL_DIR = os.path.join(DATA_DIR, "historical-data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Columns used for infrastructure (splitting, targets, evaluation) — never model features
NON_FEATURE_COLS = {'Year', 'y.status_place', 'Win Odds'}


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
def app_css():
    """Apply consistent CSS styling across all pages."""
    st.markdown(
        """
        <style>
        html, body, [class*="css"] {font-size: 12px !important;}
        h1 {font-size: 24px !important;}
        h2 {font-size: 18px !important;}
        h3, h4 {font-size: 16px !important;}
        .stButton > button {font-size: 12px !important;}
        label {font-size: 12px !important;}
        .stSelectbox label, .stSlider label, .stNumberInput label, .stMultiSelect label {font-size: 12px !important;}
        .stMarkdown {font-size: 12px !important;}
        .block {border: 1px solid #ccc; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading processed data...")
def load_processed_data(extended=False):
    """Load preprocessed race features from parquet."""
    filename = "race_features_extended.parquet" if extended else "race_features.parquet"
    filepath = os.path.join(PROCESSED_DIR, filename)
    if not os.path.exists(filepath):
        return None
    return pd.read_parquet(filepath)


@st.cache_data(show_spinner="Loading extended features...")
def load_extended_data():
    """Load extended race features (includes horse_id, dates, etc.)."""
    path = os.path.join(PROCESSED_DIR, "race_features_extended.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    for dc in ['Date', 'r_date']:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors='coerce')
    return df


@st.cache_data(show_spinner="Loading raw race data...")
def load_raw_aspx_summary(years=range(2010, 2027)):
    """Load raw ASPX results and compute summary statistics."""
    dfs = []
    for year in years:
        filepath = os.path.join(HISTORICAL_DIR, "aspx-results", f"aspx-results-{year}.csv")
        if os.path.exists(filepath):
            try:
                dfs.append(pd.read_csv(filepath, low_memory=False))
            except Exception:
                continue

    if not dfs:
        return None, None

    full_df = pd.concat(dfs, ignore_index=True)

    summary = {
        'total_rows': len(full_df),
        'unique_dates': full_df['Date'].nunique() if 'Date' in full_df.columns else 0,
        'unique_horses': full_df['Horse'].nunique() if 'Horse' in full_df.columns else 0,
        'unique_jockeys': full_df['Jockey'].nunique() if 'Jockey' in full_df.columns else 0,
        'columns': full_df.columns.tolist(),
    }

    sample_df = full_df.head(500)
    return summary, sample_df


@st.cache_data(show_spinner="Loading training data...")
def load_training_data():
    """Load data for model training."""
    path = os.path.join(PROCESSED_DIR, "race_features.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ---------------------------------------------------------------------------
# Column finders
# ---------------------------------------------------------------------------
def find_horse_col(df):
    """Find the best horse identifier column in the dataframe."""
    for col in ['horse_id', 'Horse_link', 'Horse', 'Participant']:
        if col in df.columns:
            return col
    return None


def find_date_col(df):
    """Find the best date column."""
    for col in ['Date', 'r_date']:
        if col in df.columns:
            return col
    return None


def setup_sidebar():
    """Render the shared sidebar."""
    with st.sidebar:
        st.caption("Built with Streamlit + Plotly")


# ---------------------------------------------------------------------------
# Bankroll helpers
# ---------------------------------------------------------------------------
def safe_load_parquet(path):
    """Load a parquet file if it exists, else None."""
    if os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            return None
    return None


def detect_model_prefixes():
    """Scan results/ for saved model artifacts and return list of prefixes."""
    if not os.path.isdir(RESULTS_DIR):
        return []
    prefixes = set()
    for f in os.listdir(RESULTS_DIR):
        if f.endswith("_result_df.parquet"):
            prefixes.add(f.replace("_result_df.parquet", ""))
    return sorted(prefixes)


