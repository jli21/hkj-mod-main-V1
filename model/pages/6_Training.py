"""Training page — model training with hyperparameter controls."""

import os
import json
import shutil
import time
import subprocess
import streamlit as st

from ui_helpers import (app_css, setup_sidebar, SCRIPT_DIR, PROCESSED_DIR,
                        RESULTS_DIR, NON_FEATURE_COLS, load_training_data)

app_css()
setup_sidebar()

st.header("Model Training")

train_data = load_training_data()
if train_data is None:
    st.error("Training data not found. Run `prep-data.py` first.")
    st.stop()

# ---------------------------------------------------------------------------
# Data overview
# ---------------------------------------------------------------------------
st.subheader("Data")
dc1, dc2, dc3 = st.columns(3)
with dc1:
    st.metric("Rows", f"{len(train_data):,}")
with dc2:
    st.metric("Features", f"{len(train_data.columns)}")
with dc3:
    years_available = sorted(train_data['Year'].unique())
    st.metric("Years", f"{years_available[0]}-{years_available[-1]}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.subheader("Configuration")

tab_split, tab_params, tab_features = st.tabs(["Train/Test Split", "Hyperparameters", "Features"])

with tab_split:
    col_l, col_r = st.columns(2)
    with col_l:
        years = list(range(2010, 2026))
        train_year_range = st.slider(
            "Training Years", min_value=years[0], max_value=years[-1],
            value=(2016, 2023), key="train_years",
        )
        train_years = list(range(train_year_range[0], train_year_range[1] + 1))
    with col_r:
        test_year = st.selectbox("Test Year", years, index=len(years) - 1, key="test_year")
        if test_year in train_years:
            st.warning("Overlap between train and test years.")

    race_ids = train_data.index.get_level_values(0).astype(str)
    train_count = (race_ids.str[:4].isin([str(y) for y in train_years])).sum()
    test_count = (race_ids.str[:4] == str(test_year)).sum()

    sc1, sc2 = st.columns(2)
    with sc1:
        st.caption(f"Train: {train_count:,} rows ({len(train_years)} years)")
    with sc2:
        st.caption(f"Test: {test_count:,} rows")

with tab_params:
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        learning_rate = st.number_input("Learning Rate", 0.01, 0.1, 0.02, step=0.01, key="lr")
        max_depth = st.slider("Max Depth", 3, 10, 4, key="max_depth")
        min_child_weight = st.number_input("Min Child Weight", 1, 50, 20, key="mcw")
    with col_b:
        subsample = st.slider("Subsample", 0.5, 1.0, 0.8, step=0.1, key="subsample")
        colsample_bytree = st.slider("Col Sample", 0.5, 1.0, 0.8, step=0.1, key="colsample")
        gamma = st.number_input("Gamma", 0.0, 5.0, 0.1, step=0.1, key="gamma")
    with col_c:
        reg_alpha = st.number_input("Reg Alpha (L1)", 0.0, 5.0, 1.0, step=0.1, key="reg_alpha")
        reg_lambda = st.number_input("Reg Lambda (L2)", 0.0, 10.0, 5.0, step=0.5, key="reg_lambda")
        rounds = st.number_input("Boosting Rounds", 50, 500, 300, step=10, key="rounds")
        cv_folds = st.slider("CV Folds", 3, 20, 3, key="cv_folds")

with tab_features:
    col_fl, col_fr = st.columns(2)
    with col_fl:
        include_market = st.checkbox("Include market features", value=True, key="train_include_market")
        run_market_baseline = st.checkbox("Run M0 market baseline", value=True, key="train_market_baseline")
        calibrate = st.checkbox("Isotonic calibration", value=True, key="train_calibrate",
                                help="Post-hoc calibration using last training year. Improves probability accuracy.")
    with col_fr:
        feature_cols = [c for c in train_data.columns if c not in NON_FEATURE_COLS]
        excluded_cols = st.multiselect(
            "Exclude features", feature_cols, default=[], key="exclude_cols",
        )

    with st.expander("Preview features"):
        active_features = [c for c in feature_cols if c not in excluded_cols]
        if not include_market:
            active_features = [c for c in active_features
                               if c not in ['log_implied_prob', 'market_rank', 'Implied_Prob']]
        st.caption(f"{len(active_features)} active features")
        st.code(", ".join(active_features), language="text")

# ---------------------------------------------------------------------------
# Model selection + run
# ---------------------------------------------------------------------------
st.subheader("Run")

col_model, col_topn, col_run = st.columns([2, 1, 1])
with col_model:
    models = ["Exploded Logit", "Softmax", "Offset"]
    selected_models = st.multiselect("Models", models, default=["Softmax"], key="selected_models")
    if "Offset" in selected_models:
        st.info("Offset model uses its own tuned params (lr=0.01, depth=5, L2=10, 400 rounds). "
                "Hyperparameter sliders above apply to Softmax/Exploded Logit only.")
with col_topn:
    top_n = st.number_input("Top-n (Exploded, -1=full)", -1, 10, -1, step=1, key="top_n")
with col_run:
    st.write("")  # spacer
    st.write("")
    run_button = st.button("Run Models", key="run_models_btn", type="primary", use_container_width=True)

if run_button:
    if not selected_models:
        st.error("Select at least one model.")
    else:
        if os.path.isdir(RESULTS_DIR):
            shutil.rmtree(RESULTS_DIR)
        os.makedirs(RESULTS_DIR, exist_ok=True)

        xgb_params = {
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "tree_method": "auto",
            "seed": 42,
        }
        xgb_params_str = json.dumps(xgb_params).replace('"', "'")
        train_years_str = ','.join(map(str, train_years))
        models_str = ','.join(selected_models)
        all_excluded = list(excluded_cols)
        if not include_market:
            all_excluded.extend(['log_implied_prob', 'market_rank', 'Implied_Prob'])
        exclude_str = ','.join(all_excluded)

        cmd = [
            'python', os.path.join(SCRIPT_DIR, 'trainer.py'),
            '--train_years', train_years_str,
            '--test_year', str(test_year),
            '--cv_folds', str(cv_folds),
            '--rounds', str(rounds),
            '--models', models_str,
            '--xgb_params', xgb_params_str,
            '--exclude_cols', exclude_str,
            '--top_n', str(top_n),
        ]
        if run_market_baseline:
            cmd.append('--market_baseline')
        if calibrate:
            cmd.append('--calibrate')

        output_area = st.empty()
        output_area.text("Starting training...")

        start_time = time.time()
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, cwd=SCRIPT_DIR,
        )

        logs = []
        for line in iter(process.stdout.readline, ''):
            if line:
                logs.append(line.strip())
                output_area.code("\n".join(logs[-30:]), language="text")

        stdout_rest, stderr = process.communicate()
        if stdout_rest:
            logs.extend(stdout_rest.splitlines())
        if stderr:
            st.error(f"stderr:\n{stderr}")

        elapsed = time.time() - start_time
        st.cache_data.clear()

        if process.returncode == 0:
            st.success(f"Training completed in {elapsed:.1f}s.")
            st.session_state["model_names"] = selected_models
            st.session_state["training_logs"] = "\n".join(logs)
        else:
            st.error(f"Training failed (exit code {process.returncode}).")
            st.session_state["training_logs"] = "\n".join(logs)

training_logs = st.session_state.get("training_logs", "")
if training_logs:
    with st.expander("Training Logs", expanded=False):
        st.code(training_logs, language="text")
