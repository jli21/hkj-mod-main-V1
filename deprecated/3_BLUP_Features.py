"""BLUP Features page — summary statistics and coverage."""

import pandas as pd
import streamlit as st

from ui_helpers import app_css, setup_sidebar, get_available_blup_cols

app_css()
blup_filtered, _ = setup_sidebar()

st.header("BLUP Features Analysis")

if blup_filtered is not None:
    available_blup = get_available_blup_cols(blup_filtered)

    if not available_blup:
        st.warning("No BLUP columns found in the data.")
    else:
        st.subheader("Summary Statistics")
        st.dataframe(
            blup_filtered[available_blup].describe().round(6),
            use_container_width=True,
        )

        with st.expander("Sample Rows", expanded=False):
            st.dataframe(blup_filtered[available_blup].head(50),
                         use_container_width=True)

        st.subheader("Feature Coverage")
        coverage = []
        for col in available_blup:
            total = len(blup_filtered)
            non_zero = (blup_filtered[col] != 0).sum()
            non_null = blup_filtered[col].notna().sum()
            coverage.append({
                'Feature': col,
                'Non-Null': f"{non_null:,}",
                'Non-Zero': f"{non_zero:,}",
                'Coverage %': f"{non_zero / total * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(coverage), use_container_width=True, hide_index=True)
else:
    st.info("No BLUP data available. Run `prep_data.py` to generate.")
