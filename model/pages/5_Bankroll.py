"""Bankroll page — simulation results, log loss comparison, feature importance."""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import log_loss as sk_log_loss

from ui_helpers import (app_css, setup_sidebar, RESULTS_DIR,
                        safe_load_parquet, detect_model_prefixes)

try:
    from betting import simulate_bankroll
except ImportError:
    simulate_bankroll = None

app_css()
setup_sidebar()


@st.cache_data(show_spinner="Running bankroll simulation...")
def _cached_simulation(prefix, alpha, initial_bankroll):
    """Load result artifacts and run simulate_bankroll. Cached."""
    if simulate_bankroll is None:
        return None, None, None, None, None

    base = os.path.join(RESULTS_DIR, prefix)
    result_df = safe_load_parquet(base + "_result_df.parquet")
    y_test_win_df = safe_load_parquet(base + "_y_test_win.parquet")
    odds_df = safe_load_parquet(base + "_odds.parquet")
    dividends_df = safe_load_parquet(base + "_dividends.parquet")

    if result_df is None or y_test_win_df is None or odds_df is None:
        return None, None, None, None, None

    if isinstance(y_test_win_df, pd.DataFrame):
        y_series = y_test_win_df.iloc[:, 0]
    else:
        y_series = y_test_win_df

    idx_names = list(y_series.index.names)
    ret = y_series.reset_index().rename(columns={y_series.name: 'y.status_win'})
    probs_flat = result_df[['Y_prob']].reset_index()
    ret = ret.merge(probs_flat, on=idx_names, how='inner')
    ret = ret.rename(columns={'Y_prob': 'pred_prob'})

    if dividends_df is not None:
        try:
            div_flat = dividends_df.reset_index()
            merge_on = [c for c in idx_names if c in div_flat.columns]
            if merge_on:
                ret = ret.merge(div_flat, on=merge_on, how='left', suffixes=('', '_div'))
            else:
                ret = ret.merge(div_flat, left_on=idx_names[0], right_index=True, how='left', suffixes=('', '_div'))
        except Exception:
            pass

    try:
        odds_flat = odds_df.reset_index() if isinstance(odds_df, pd.DataFrame) else odds_df.to_frame(name="Win Odds").reset_index()
        merge_on = [c for c in idx_names if c in odds_flat.columns]
        if merge_on:
            ret = ret.merge(odds_flat, on=merge_on, how='left', suffixes=('', '_odds'))
        else:
            ret = ret.merge(odds_flat, left_on=idx_names[0], right_index=True, how='left', suffixes=('', '_odds'))
    except Exception:
        pass

    if 'race_id' not in ret.columns and idx_names[0] in ret.columns:
        ret = ret.rename(columns={idx_names[0]: 'race_id'})

    race_df, bet_df, bankroll_without, bankroll_with = simulate_bankroll(
        ret, alpha, initial_bankroll=initial_bankroll,
        probability_col='pred_prob', kelly_odds_col='Win Odds',
        payoff_col='Returns', result_col='y.status_win',
    )
    return race_df, bet_df, bankroll_without, bankroll_with, result_df


# ---------------------------------------------------------------------------
# Page content
# ---------------------------------------------------------------------------
st.header("Bankroll Simulation")

model_prefixes = detect_model_prefixes()

if simulate_bankroll is None:
    st.error("Could not import `simulate_bankroll` from betting.py.")
elif not model_prefixes:
    st.info(
        "No model results found in `results/`. "
        "Run training first (via the Training page or `trainer.py`)."
    )
else:
    for prefix in model_prefixes:
        display_name = prefix.replace("_", " ").title()
        st.subheader(f"Model: {display_name}")

        col_a, col_b = st.columns(2)
        with col_a:
            alpha = st.number_input(
                "Kelly Fraction (alpha)", 0.01, 1.0, 0.10,
                step=0.01, key=f"alpha_{prefix}",
            )
        with col_b:
            initial_bankroll = st.number_input(
                "Initial Bankroll (HKD)", 10_000, 100_000_000, 1_000_000,
                step=100_000, key=f"bankroll_{prefix}",
            )

        race_df, bet_df, bw, br, result_df = _cached_simulation(
            prefix, alpha, initial_bankroll,
        )

        if race_df is None:
            st.warning(f"Missing artifacts for **{display_name}**.")
            continue

        # --- Bankroll trajectory ---
        race_dates = pd.to_datetime(
            race_df['race_id'].astype(str).str[:8], format="%Y%m%d", errors='coerce'
        )
        n = len(race_dates)
        x = list(range(n))

        bw_vals = bw.values.astype(float) if bw is not None else np.full(n, np.nan)
        br_vals = br.values.astype(float) if br is not None else np.full(n, np.nan)

        use_webgl = n > 3000
        scatter_cls = go.Scattergl if use_webgl else go.Scatter

        fig_bankroll = go.Figure()
        fig_bankroll.add_trace(scatter_cls(
            x=x, y=bw_vals, mode='lines', name='Without Rebate',
            line=dict(width=2.5, color='#636EFA'),
        ))
        fig_bankroll.add_trace(scatter_cls(
            x=x, y=br_vals, mode='lines', name='With Rebate',
            line=dict(width=2, color='#00CC96'),
            opacity=0.7,
        ))
        fig_bankroll.add_hline(
            y=initial_bankroll, line_dash="dash", line_color="rgba(255,255,255,0.3)",
            annotation_text="Initial", annotation_position="top left",
        )

        n_ticks = 12
        step = max(1, n // n_ticks)
        tick_locs = x[::step]
        tick_text = race_dates.dt.strftime("%Y-%m").iloc[::step].tolist()

        fig_bankroll.update_layout(
            xaxis=dict(tickmode="array", tickvals=tick_locs, ticktext=tick_text,
                       title="", gridcolor='rgba(128,128,128,0.15)'),
            yaxis=dict(title="Bankroll (HKD)", gridcolor='rgba(128,128,128,0.15)',
                       tickformat='$,.0f'),
            height=450,
            margin=dict(l=60, r=20, t=30, b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig_bankroll, use_container_width=True)

        # --- Summary metrics ---
        final_bw = bw_vals[-1] if len(bw_vals) > 0 else initial_bankroll
        final_br = br_vals[-1] if len(br_vals) > 0 else initial_bankroll
        n_races = len(race_df)

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.metric("Races", f"{n_races:,}")
        with mc2:
            st.metric("Final (no rebate)", f"${final_bw:,.0f}",
                      delta=f"{(final_bw / initial_bankroll - 1) * 100:+.2f}%")
        with mc3:
            st.metric("Final (with rebate)", f"${final_br:,.0f}",
                      delta=f"{(final_br / initial_bankroll - 1) * 100:+.2f}%")

        # --- Log Loss Comparison ---
        if bet_df is not None and 'pred_prob' in bet_df.columns and 'y.status_win' in bet_df.columns:
            valid = bet_df.dropna(subset=['pred_prob', 'y.status_win'])
            if 'Win Odds' in valid.columns and len(valid) > 0:
                market_prob = 1.0 / valid['Win Odds']
                market_prob = market_prob / valid.groupby('race_id')['Win Odds'].transform(
                    lambda x: (1.0 / x).sum()
                )
                ll_model = sk_log_loss(valid['y.status_win'], valid['pred_prob'], labels=[0, 1])
                ll_market = sk_log_loss(valid['y.status_win'], market_prob.values, labels=[0, 1])
                delta_ll = ll_model - ll_market

                lc1, lc2, lc3 = st.columns(3)
                with lc1:
                    st.metric("Model Log Loss", f"{ll_model:.5f}")
                with lc2:
                    st.metric("Market Log Loss", f"{ll_market:.5f}")
                with lc3:
                    color = "normal" if delta_ll < 0 else "inverse"
                    st.metric("Model vs Market", f"{delta_ll:+.5f}",
                              delta=f"{'beats' if delta_ll < 0 else 'worse'}", delta_color=color)

        # --- Feature Importance (full width horizontal bar) ---
        imp_df = safe_load_parquet(
            os.path.join(RESULTS_DIR, f"{prefix}_feature_importance.parquet")
        )
        if imp_df is not None and len(imp_df) > 0:
            with st.expander("Feature Importance", expanded=True):
                if 'feature' in imp_df.columns and 'importance' in imp_df.columns:
                    plot_df = imp_df.sort_values('importance', ascending=True).tail(25)
                    table_df = imp_df.sort_values('importance', ascending=False)
                else:
                    num_col = imp_df.select_dtypes(include='number').columns[0]
                    str_col = imp_df.select_dtypes(include='object').columns[0] if len(imp_df.select_dtypes(include='object').columns) > 0 else imp_df.columns[0]
                    plot_df = imp_df.sort_values(num_col, ascending=True).tail(25)
                    plot_df = plot_df.rename(columns={str_col: 'feature', num_col: 'importance'})
                    table_df = imp_df.sort_values(num_col, ascending=False).rename(
                        columns={str_col: 'feature', num_col: 'importance'})

                col_chart, col_table = st.columns([3, 1])
                with col_chart:
                    fig_imp = go.Figure(go.Bar(
                        x=plot_df['importance'].values,
                        y=plot_df['feature'].values,
                        orientation='h',
                        marker_color='#636EFA',
                    ))
                    fig_imp.update_layout(
                        height=max(400, len(plot_df) * 22),
                        margin=dict(l=10, r=20, t=10, b=10),
                        xaxis=dict(title="Gain", gridcolor='rgba(128,128,128,0.15)'),
                        yaxis=dict(title=""),
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                with col_table:
                    st.dataframe(
                        table_df[['feature', 'importance']].round(1),
                        use_container_width=True, hide_index=True,
                        height=max(400, len(plot_df) * 22),
                    )

        # --- Detailed tables ---
        with st.expander("Race-Level Results"):
            st.dataframe(race_df, use_container_width=True)
        with st.expander("Bet-Level Details"):
            st.dataframe(bet_df, use_container_width=True)

        st.markdown("---")
