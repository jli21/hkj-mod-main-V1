"""Raw Data page — browse raw ASPX results and processed features."""

import streamlit as st

from ui_helpers import (app_css, setup_sidebar, load_raw_aspx_summary,
                        load_processed_data)

app_css()
setup_sidebar()

st.header("Raw Input Data")
st.markdown(
    "Sample of raw ASPX race results before preprocessing. "
    "Full data is loaded for metric computation on the home page."
)

summary, raw_sample = load_raw_aspx_summary()

if raw_sample is not None:
    all_cols = raw_sample.columns.tolist()
    selected_cols = st.multiselect(
        "Select columns to display",
        all_cols,
        default=all_cols[:12] if len(all_cols) > 12 else all_cols,
        key="raw_cols",
    )
    if selected_cols:
        st.dataframe(raw_sample[selected_cols], use_container_width=True)
else:
    st.warning("No raw data files found.")

with st.expander("Processed Data Preview"):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Basic Features**")
        proc = load_processed_data(extended=False)
        if proc is not None:
            st.write(f"{len(proc):,} rows x {len(proc.columns)} cols")
            st.dataframe(proc.head(20), use_container_width=True)
        else:
            st.info("Not available.")
    with col2:
        st.markdown("**Extended Features**")
        ext = load_processed_data(extended=True)
        if ext is not None:
            st.write(f"{len(ext):,} rows x {len(ext.columns)} cols")
            st.dataframe(ext.head(20), use_container_width=True)
        else:
            st.info("Not available.")
