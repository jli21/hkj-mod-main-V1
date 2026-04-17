"""Horse Analysis page — race history and edge features per horse."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui_helpers import (app_css, setup_sidebar, load_extended_data,
                        find_horse_col, find_date_col)

app_css()
setup_sidebar()

st.header("Horse Performance Analysis")

ext = load_extended_data()
if ext is None:
    st.info("No extended features found. Run `prep-data.py` first.")
    st.stop()

horse_col = find_horse_col(ext)
date_col = find_date_col(ext)

if horse_col is None:
    st.error("No horse identifier column found in extended features.")
    st.stop()

unique_horses = sorted(ext[horse_col].dropna().unique())

col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    selected_horse = st.selectbox("Select Horse", unique_horses, key="horse_select")
with col_sel2:
    horse_df = ext[ext[horse_col] == selected_horse].copy()
    st.metric("Races Found", len(horse_df))

if len(horse_df) == 0:
    st.warning(f"No data for {selected_horse}.")
    st.stop()

if date_col:
    horse_df = horse_df.sort_values(date_col)

# ---- Race History Timeline ----
st.subheader("Race History")

if 'Win Odds' in horse_df.columns and date_col:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=horse_df[date_col], y=horse_df['Win Odds'],
        mode='lines+markers', name='Win Odds',
        line=dict(color='#636EFA', width=2),
        marker=dict(size=6),
    ))

    if 'y.status_place' in horse_df.columns:
        wins = horse_df[horse_df['y.status_place'] == 1]
        if len(wins) > 0:
            fig.add_trace(go.Scatter(
                x=wins[date_col], y=wins['Win Odds'],
                mode='markers', name='Win',
                marker=dict(size=12, color='#2ecc71', symbol='star'),
            ))

    fig.update_layout(
        title=f"{selected_horse} — Win Odds Over Time",
        xaxis_title="Date", yaxis_title="Win Odds",
        height=400, hovermode='x unified',
    )
    st.plotly_chart(fig, use_container_width=True)

# ---- Performance Summary ----
if 'y.status_place' in horse_df.columns:
    mc1, mc2, mc3, mc4 = st.columns(4)
    places = horse_df['y.status_place']
    with mc1:
        st.metric("Wins", int((places == 1).sum()))
    with mc2:
        st.metric("Top 3", int((places <= 3).sum()))
    with mc3:
        st.metric("Avg Finish", f"{places.mean():.1f}")
    with mc4:
        if 'Win Odds' in horse_df.columns:
            st.metric("Avg Odds", f"{horse_df['Win Odds'].mean():.1f}")

# ---- Edge Features Over Time ----
edge_cols = [c for c in [
    'class_change', 'weight_change', 'recent_form', 'days_since',
    'trainer_moves', 'career_beat_odds', 'wt_z',
] if c in horse_df.columns]

if edge_cols and date_col:
    st.subheader("Edge Features Over Time")
    selected_edge = st.multiselect("Select features", edge_cols, default=edge_cols[:3])

    if selected_edge:
        fig_edge = go.Figure()
        for col in selected_edge:
            fig_edge.add_trace(go.Scatter(
                x=horse_df[date_col], y=horse_df[col],
                mode='lines+markers', name=col,
            ))
        fig_edge.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_edge, use_container_width=True)

# ---- Race Details Table ----
with st.expander("Race-by-Race Details"):
    display_cols = [c for c in [
        date_col, 'y.status_place', 'Win Odds', 'Act_Wt', 'Draw',
        'class_change', 'recent_form', 'days_since', 'trainer_moves',
        'TimeSec',
    ] if c and c in horse_df.columns]
    st.dataframe(
        horse_df[display_cols].round(2),
        use_container_width=True, hide_index=True,
    )
