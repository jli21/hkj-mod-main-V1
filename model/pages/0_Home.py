"""Home page — overview metrics and status."""

import streamlit as st

from ui_helpers import app_css, load_raw_aspx_summary, setup_sidebar, load_processed_data

app_css()
setup_sidebar()

st.title("HKJC Racing Model Dashboard")

summary, raw_sample = load_raw_aspx_summary()

if summary:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Race Dates", f"{summary['unique_dates']:,}")
    with c2:
        st.metric("Unique Horses", f"{summary['unique_horses']:,}")
    with c3:
        st.metric("Unique Jockeys", f"{summary['unique_jockeys']:,}")
    with c4:
        st.metric("Total Rows", f"{summary['total_rows']:,}")
else:
    st.info("No raw data files found. Place ASPX CSV files in data/historical-data/aspx-results/.")

st.markdown("---")

processed = load_processed_data()
if processed is not None:
    st.success(f"Processed features loaded: {len(processed):,} rows, {len(processed.columns)} features.")
else:
    st.warning("No processed features found. Run `prep-data.py` to generate.")

st.markdown("Navigate between pages using the sidebar menu.")
