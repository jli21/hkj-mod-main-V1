"""Race Inspector — per-race details: runners, odds, finish positions, edge features."""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from ui_helpers import (app_css, setup_sidebar, load_extended_data, find_date_col)

app_css()
setup_sidebar()

st.header("Race Inspector")

ext = load_extended_data()
if ext is None:
    st.info("No extended features found. Run `prep-data.py` first.")
    st.stop()

date_col = find_date_col(ext)

# --- Race selector ---
race_ids = ext.index.get_level_values(0).unique()

if date_col and date_col in ext.columns:
    race_dates = ext.groupby(level=0)[date_col].first()
    race_labels = {rid: f"{rid}  ({race_dates.loc[rid].strftime('%Y-%m-%d')})"
                   for rid in race_ids if rid in race_dates.index}
    sorted_ids = race_dates.sort_values(ascending=False).index.tolist()
else:
    race_labels = {rid: str(rid) for rid in race_ids}
    sorted_ids = sorted(race_ids, reverse=True)

selected_race = st.selectbox(
    "Select Race",
    sorted_ids,
    format_func=lambda x: race_labels.get(x, str(x)),
    key="race_inspector_select",
)

race_df = ext.loc[[selected_race]].copy()
n_runners = len(race_df)
st.caption(f"{n_runners} runners")

# --- Results table ---
horse_col = None
for c in ['horse_id', 'Horse', 'Name']:
    if c in race_df.columns:
        horse_col = c
        break

horse_no = race_df.index.get_level_values(1).astype(str)

result = pd.DataFrame({
    'Horse No': horse_no.values,
    'Horse': race_df[horse_col].values if horse_col else horse_no.values,
})

if 'y.status_place' in race_df.columns:
    result['Finish'] = race_df['y.status_place'].values

if 'Win Odds' in race_df.columns:
    result['Win Odds'] = race_df['Win Odds'].values
    result['Implied Prob'] = (1.0 / race_df['Win Odds'].values)
    result['Implied Prob'] = result['Implied Prob'] / result['Implied Prob'].sum()

if 'Implied_Prob' in race_df.columns:
    result['Model Implied'] = race_df['Implied_Prob'].values

if 'TimeSec' in race_df.columns:
    result['Time (s)'] = race_df['TimeSec'].values

if 'Finish' in result.columns:
    result = result.sort_values('Finish')

# --- Bar chart: Win Odds by horse ---
if 'Win Odds' in result.columns:
    st.subheader("Win Odds")
    fig = go.Figure()
    colors = ['#2ecc71' if f == 1 else '#636EFA'
              for f in result.get('Finish', [0]*len(result))]
    fig.add_trace(go.Bar(
        y=result['Horse No'],
        x=result['Win Odds'],
        marker_color=colors,
        orientation='h',
    ))
    fig.update_layout(
        yaxis_title='Horse No',
        xaxis_title='Win Odds',
        height=max(350, n_runners * 35),
        yaxis=dict(autorange='reversed'),
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Ranking analysis ---
if 'Finish' in result.columns and 'Win Odds' in result.columns:
    st.subheader("Market Ranking vs Actual")

    result['Market Rank'] = result['Win Odds'].rank(method='min').astype(int)
    rank_df = result[['Horse No', 'Horse', 'Finish', 'Market Rank']].copy()
    rank_df['Rank Diff'] = rank_df['Finish'] - rank_df['Market Rank']

    def rank_color(val):
        if abs(val) <= 1:
            return 'color: #2ecc71'
        elif abs(val) <= 3:
            return 'color: #f39c12'
        else:
            return 'color: #e74c3c'

    styled = rank_df.style.applymap(rank_color, subset=['Rank Diff'])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    mc1, mc2, mc3 = st.columns(3)
    winner_market_rank = rank_df.loc[rank_df['Finish'] == 1, 'Market Rank']
    top3_actual = set(rank_df[rank_df['Finish'] <= 3]['Horse No'])
    top3_market = set(rank_df[rank_df['Market Rank'] <= 3]['Horse No'])
    top3_overlap = len(top3_actual & top3_market)

    with mc1:
        st.metric("Winner Market Rank",
                  int(winner_market_rank.iloc[0]) if len(winner_market_rank) > 0 else "--")
    with mc2:
        st.metric("Top-3 Overlap", f"{top3_overlap}/3")
    with mc3:
        mae_rank = rank_df['Rank Diff'].abs().mean()
        st.metric("Mean Rank Error", f"{mae_rank:.1f}")

# --- Edge features for this race ---
edge_cols = [c for c in [
    'class_change', 'weight_change', 'recent_form', 'days_since',
    'trainer_moves', 'career_beat_odds', 'wt_z', 'trainer_track_spec',
    'bt_avg_early', 'bt_last_behind',
] if c in race_df.columns]

if edge_cols:
    with st.expander("Edge Features"):
        edge_display = race_df[edge_cols].copy()
        edge_display.insert(0, 'Horse No', horse_no.values)
        st.dataframe(edge_display.round(3), use_container_width=True, hide_index=True)

# --- Full table ---
with st.expander("Full Details"):
    display = result.copy()
    for c in result.select_dtypes(include='float').columns:
        display[c] = display[c].round(2)
    st.dataframe(display, use_container_width=True, hide_index=True)
