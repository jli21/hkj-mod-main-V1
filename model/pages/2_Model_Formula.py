"""Model Overview page — current model architecture and features."""

import pandas as pd
import streamlit as st

from ui_helpers import app_css, setup_sidebar

app_css()
setup_sidebar()

st.header("Model Overview")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("Architecture")
    st.markdown("""
    **XGBoost Softmax / Exploded Logit**

    Race-level grouped model that directly estimates win probabilities.
    Each race is a group; the model learns per-horse logits that are
    softmaxed within each race to produce calibrated probabilities.

    ```
    Input:   Market features + edge features per runner
    Output:  P(win | race) per horse, summing to 1 within race
    ```

    **Post-hoc calibration:**
    - Binned temperature scaling (by field size and surface)
    - Optional isotonic calibration on held-out year

    **Betting:**
    - Multinomial Kelly criterion with analytical gradients
    - Fractional Kelly (alpha = 0.01)
    - Edge filtering + optional uncertainty damping
    """)

    st.subheader("Best Config")
    st.code("""
params = {
    'learning_rate': 0.02,
    'max_depth': 4,
    'min_child_weight': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 1.0,
    'reg_lambda': 5.0,
}
rounds = 300
    """, language="python")

with col_right:
    st.subheader("Market Features")
    market_features = pd.DataFrame({
        'Feature': ['Implied_Prob', 'log_implied_prob', 'Racecourse', 'Track_Turf', 'Draw', 'Act_Wt'],
        'Type': ['Continuous', 'Continuous', 'Binary', 'Binary', 'Continuous', 'Continuous'],
        'Description': [
            'Normalized implied probability from odds',
            'Log of implied probability',
            'Sha Tin=1, Happy Valley=0',
            'Turf=1, AWT=0',
            'Starting barrier position',
            'Actual weight carried (lbs)',
        ],
    })
    st.dataframe(market_features, use_container_width=True, hide_index=True)

    st.subheader("Edge Features (18)")
    edge_features = pd.DataFrame({
        'Feature': [
            'class_change', 'class_drop_odds_stable', 'weight_change', 'wt_z',
            'recent_form', 'trainer_track_spec',
            'draw_outside_ST', 'draw_inside_HV',
            'streak_good', 'prev_win', 'fav_field_size',
            'days_since', 'trainer_moves', 'career_beat_odds',
            'field_form_cv',
            'bt_avg_early', 'bt_last_behind', 'bt_recent_behind',
        ],
        'Description': [
            'Class drop/rise vs previous race',
            'Class drop + stable odds interaction',
            'Weight change vs previous race',
            'Weight z-score within race',
            'Avg finish position last 3 races',
            'Trainer win-rate delta (track vs overall)',
            'Outside draw at Sha Tin (>=10)',
            'Inside draw at Happy Valley (<=3)',
            'Consecutive top-3 finishes',
            'Won previous race',
            'Favourite x field size',
            'Days since last race',
            'Count of trainer moves (class/weight/jockey)',
            'Career outperformance vs market',
            'Field form coefficient of variation',
            'Barrier trial avg early position',
            'Barrier trial last time behind winner',
            'Barrier trial recent avg time behind',
        ],
    })
    st.dataframe(edge_features, use_container_width=True, hide_index=True)
