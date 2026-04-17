"""Diagnostics page — shrinkage-to-market, edge buckets, model comparison."""

import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from ui_helpers import (app_css, setup_sidebar,
                        RESULTS_DIR, safe_load_parquet, detect_model_prefixes)

app_css()
setup_sidebar()

st.header("Model Diagnostics")

# =====================================================================
# Section 1: Shrinkage-to-Market
# =====================================================================
st.subheader("Shrinkage-to-Market Sweep")
st.markdown(
    "Blends model probabilities with market probabilities at different weights. "
    "If bankroll improves when shrinking toward market, the model is **overconfident**."
)

model_prefixes = detect_model_prefixes()
shrinkage_found = False

if model_prefixes:
    fig_shrink = go.Figure()
    for prefix in model_prefixes:
        if prefix.startswith('m0_'):
            continue
        shrink_path = os.path.join(RESULTS_DIR, f"{prefix}_shrinkage.parquet")
        shrink_df = safe_load_parquet(shrink_path)
        if shrink_df is not None and len(shrink_df) > 0:
            shrinkage_found = True
            display_name = prefix.replace("_", " ").title()
            fig_shrink.add_trace(go.Scatter(
                x=shrink_df['model_weight'],
                y=shrink_df['final_bankroll_with_rebate'],
                mode='lines+markers',
                name=display_name,
                marker=dict(size=8),
            ))
            best_idx = shrink_df['final_bankroll_with_rebate'].idxmax()
            best_row = shrink_df.loc[best_idx]
            if best_row['model_weight'] < 1.0:
                st.warning(
                    f"**{display_name}**: Optimal blend is "
                    f"{best_row['model_weight']:.0%} model / {best_row['market_weight']:.0%} market "
                    f"-- model may be overconfident."
                )

    if shrinkage_found:
        fig_shrink.update_layout(
            xaxis_title="Model Weight (0 = pure market, 1 = pure model)",
            yaxis_title="Final Bankroll (with rebate)",
            height=400,
        )
        st.plotly_chart(fig_shrink, use_container_width=True)

        for prefix in model_prefixes:
            if prefix.startswith('m0_'):
                continue
            shrink_df = safe_load_parquet(os.path.join(RESULTS_DIR, f"{prefix}_shrinkage.parquet"))
            if shrink_df is not None:
                with st.expander(f"{prefix.replace('_', ' ').title()} -- Shrinkage Details"):
                    st.dataframe(shrink_df.round(4), use_container_width=True, hide_index=True)

if not shrinkage_found:
    st.info("No shrinkage data found. Run training with evaluation enabled.")

# =====================================================================
# Section 2: Edge Bucket Realization
# =====================================================================
st.subheader("Edge Bucket Realization")
st.markdown(
    "Bets grouped by estimated edge. If higher edge does not correspond to "
    "better realized returns, the model signal is not actionable."
)

edge_found = False
if model_prefixes:
    for prefix in model_prefixes:
        if prefix.startswith('m0_'):
            continue
        edge_path = os.path.join(RESULTS_DIR, f"{prefix}_edge_buckets.parquet")
        edge_df = safe_load_parquet(edge_path)
        if edge_df is not None and len(edge_df) > 0:
            edge_found = True
            display_name = prefix.replace("_", " ").title()
            st.markdown(f"**{display_name}**")

            fig_edge = px.bar(
                edge_df, x='bucket', y='realized_roi',
                text='n_bets', title=f"Realized ROI by Edge Bucket -- {display_name}",
                labels={'bucket': 'Estimated Edge', 'realized_roi': 'Realized ROI'},
            )
            fig_edge.add_hline(y=0, line_dash="dash", line_color="red")
            fig_edge.update_traces(textposition='outside', texttemplate='%{text} bets')
            fig_edge.update_layout(height=350)
            st.plotly_chart(fig_edge, use_container_width=True)

if not edge_found:
    st.info("No edge bucket data found. Run training with evaluation enabled.")

# =====================================================================
# Section 3: Model Comparison
# =====================================================================
st.subheader("Model Comparison")

metrics_rows = []
if model_prefixes:
    for prefix in model_prefixes:
        metrics_path = os.path.join(RESULTS_DIR, f"{prefix}_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                m = json.load(f)
            row = {'Model': prefix.replace('_', ' ').title()}
            row.update({k: v for k, v in m.get('probability', {}).items() if v is not None})
            if not prefix.startswith('m0_'):
                row.update({k: v for k, v in m.get('bankroll', {}).items() if v is not None})
                row.update({k: v for k, v in m.get('betting', {}).items() if v is not None})
            metrics_rows.append(row)

if metrics_rows:
    metrics_df = pd.DataFrame(metrics_rows)
    fmt_cols = {c: '{:.4f}' for c in metrics_df.columns if metrics_df[c].dtype == float}
    st.dataframe(metrics_df.style.format(fmt_cols, na_rep='--'), use_container_width=True, hide_index=True)
else:
    st.info("No model metrics found. Run training to generate.")
