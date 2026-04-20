import argparse
import json
import sys
import os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from model import split_data
from model_classifier import ExplodedLogit, SoftmaxModel, OffsetModel
from betting import simulate_bankroll
from evaluation import shrinkage_sweep, edge_bucket_analysis, comprehensive_metrics

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from utilities import conv_dict, load_pool_dividends


def _build_eval_frame(Y_test, result_df, odds, prob_col='Y_prob'):
    """Build flat evaluation DataFrame from model outputs."""
    Y_test_win = (Y_test == 1).astype(int).rename('y.status_win')
    ret = Y_test_win.reset_index()

    result_flat = result_df[[prob_col]].reset_index()
    ret = ret.merge(result_flat, on=list(Y_test.index.names), how='inner')
    ret = ret.rename(columns={prob_col: 'pred_prob'})

    dividends = conv_dict(load_pool_dividends())
    dividends_df = dividends.reset_index() if dividends.index.names != [None] else dividends.copy()
    race_id_name = Y_test.index.names[0]
    if race_id_name in dividends_df.columns and race_id_name in ret.columns:
        ret = ret.merge(dividends_df, on=race_id_name, how='left', suffixes=('', '_div'))
    else:
        ret = ret.merge(dividends_df, left_on=race_id_name, right_index=True, how='left', suffixes=('', '_div'))

    odds_flat = odds.reset_index()
    ret = ret.merge(odds_flat, on=list(Y_test.index.names), how='left', suffixes=('', '_odds'))

    if 'race_id' not in ret.columns and race_id_name in ret.columns:
        ret = ret.rename(columns={race_id_name: 'race_id'})

    return ret, Y_test_win, dividends_df


def _run_evaluation(ret, alpha, initial_bankroll, prefix):
    """Run simulation + evaluation diagnostics and save artifacts."""
    race_df, bet_df, bw, br = simulate_bankroll(
        ret, alpha, initial_bankroll=initial_bankroll,
        probability_col='pred_prob', kelly_odds_col='Win Odds',
        payoff_col='Returns', result_col='y.status_win',
    )

    # Comprehensive metrics
    metrics = comprehensive_metrics(
        ret['pred_prob'], ret['y.status_win'], ret['Win Odds'],
        race_df, bet_df, bw, br, initial_bankroll,
    )
    with open(f'results/{prefix}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Shrinkage-to-market sweep
    try:
        shrink_df = shrinkage_sweep(
            ret, alpha, initial_bankroll=initial_bankroll,
        )
        shrink_df.to_parquet(f'results/{prefix}_shrinkage.parquet')
        print(f"\n  Shrinkage-to-market sweep:")
        for _, row in shrink_df.iterrows():
            print(f"    Model {row['model_weight']:.0%} / Market {row['market_weight']:.0%}: "
                  f"bankroll=${row['final_bankroll_with_rebate']:,.0f}, "
                  f"log_loss={row['log_loss']:.4f}" if row['log_loss'] else "")
    except Exception as e:
        print(f"  Shrinkage sweep failed: {e}")

    # Edge bucket analysis
    try:
        edge_df = edge_bucket_analysis(bet_df)
        if len(edge_df) > 0:
            edge_df.to_parquet(f'results/{prefix}_edge_buckets.parquet')
            print(f"\n  Edge bucket realization:")
            for _, row in edge_df.iterrows():
                roi_str = f"{row['realized_roi']:+.2%}" if row['realized_roi'] is not None else "N/A"
                print(f"    {row['bucket']}: {row['n_bets']} bets, ROI={roi_str}")
    except Exception as e:
        print(f"  Edge bucket analysis failed: {e}")

    # Print summary
    pm = metrics['probability']
    bm = metrics['bankroll']
    print(f"\n  Metrics: log_loss={pm.get('log_loss', 'N/A')}, "
          f"brier={pm.get('brier', 'N/A')}, "
          f"cal_slope={pm.get('calibration_slope', 'N/A')}")
    print(f"  Bankroll: no_rebate=${bm['final_no_rebate']:,.0f} ({bm['roi_no_rebate']:+.2%}), "
          f"with_rebate=${bm['final_with_rebate']:,.0f} ({bm['roi_with_rebate']:+.2%}), "
          f"max_dd={bm['max_drawdown']:.2%}")

    return race_df, bet_df, bw, br


def main():
    parser = argparse.ArgumentParser(description="Train horse racing models")
    parser.add_argument('--train_years', type=str, required=True, help="Comma-separated training years")
    parser.add_argument('--test_year', type=int, required=True, help="Testing year")
    parser.add_argument('--alpha', type=float, default=0.01, help="Kelly fraction alpha")
    parser.add_argument('--cv_folds', type=int, default=3, help="Cross-validation folds")
    parser.add_argument('--rounds', type=int, default=300, help="Boosting rounds")
    parser.add_argument('--models', type=str, default="Exploded Logit,Softmax", help="Comma-separated models (Exploded Logit, Softmax, Offset)")
    parser.add_argument('--xgb_params', type=str, default="{'learning_rate':0.02,'max_depth':4,'min_child_weight':20,'subsample':0.8,'colsample_bytree':0.8,'gamma':0.1,'reg_alpha':1.0,'reg_lambda':5.0,'tree_method':'auto','nthread':10,'seed':42}", help="XGBoost params")
    parser.add_argument('--exclude_cols', type=str, default="", help="Comma-separated columns to exclude")
    parser.add_argument('--top_n', type=int, default=-1, help="Top-n truncation for Exploded Logit")
    parser.add_argument('--use_blups', action='store_true', help="Use BLUP features")
    parser.add_argument('--market_baseline', action='store_true', help="Also generate M0 market-only baseline")
    parser.add_argument('--calibrate', action='store_true', help="Apply isotonic calibration (uses last training year as calibration set)")
    parser.add_argument('--bag_seeds', type=str, default="",
                        help="Comma-separated seeds for Offset bagging, e.g. '7,17,42,123,256'. "
                             "When set, trains one Offset per seed and averages predictions. "
                             "Applies only to Offset model.")

    args = parser.parse_args()

    train_years_list = list(map(int, args.train_years.split(',')))
    models_list = args.models.split(',')
    xgb_params_dict = eval(args.xgb_params)
    initial_bankroll = 10_000_000.0

    os.makedirs('results', exist_ok=True)

    # Load data
    if args.use_blups:
        data_file = "../data/processed/race_features_with_blups.parquet"
        print(f"Loading BLUP-enhanced features from {data_file}")
    else:
        data_file = "../data/processed/race_features.parquet"
        print(f"Loading basic features from {data_file}")

    df = pd.read_parquet(data_file)
    print(f"Loaded {len(df)} rows with {len(df.columns)} columns")

    blup_cols = ['participant_re_intercept', 'participant_re_slope', 'rider_resid_mean_shrunk', 'PPM_entry']
    present_blups = [col for col in blup_cols if col in df.columns]
    if present_blups:
        print(f"BLUP features present: {', '.join(present_blups)}")

    market_cols = ['log_implied_prob', 'market_rank', 'Implied_Prob']
    present_market = [col for col in market_cols if col in df.columns]
    if present_market:
        print(f"Market features present: {', '.join(present_market)}")

    if args.exclude_cols:
        exclude_list = [c for c in args.exclude_cols.split(',') if c]
        df = df.drop(columns=exclude_list, errors='ignore')
        print(f"Excluded columns: {', '.join(exclude_list)}")

    odds = df['Win Odds']
    df = df.drop(columns=['Win Odds'])

    X_train, X_test, Y_train, Y_test = split_data(
        df, train_years_list, [args.test_year], 'y.status_place',
        remove_ent=True, shuffle=False
    )

    print(f"\nTrain: {len(X_train):,} rows, Test: {len(X_test):,} rows")
    print(f"Features: {list(X_train.columns)}")

    # ── M0: Market-only baseline ──
    if args.market_baseline:
        print("\n" + "=" * 60)
        print("M0: Market-Only Baseline")
        print("=" * 60)

        market_prob = (1.0 / odds.loc[Y_test.index])
        market_prob = market_prob / market_prob.groupby(level=0).transform('sum')
        market_result_df = pd.DataFrame({'Y_prob': market_prob}, index=Y_test.index)

        ret_m0, Y_test_win_m0, dividends_df_m0 = _build_eval_frame(
            Y_test, market_result_df, odds,
        )
        _run_evaluation(ret_m0, args.alpha, initial_bankroll, 'm0_market')

        market_result_df.to_parquet('results/m0_market_result_df.parquet')
        Y_test_win_m0.to_frame().to_parquet('results/m0_market_y_test_win.parquet')
        odds.loc[Y_test.index].reset_index().to_parquet('results/m0_market_odds.parquet')
        dividends_df_m0.to_parquet('results/m0_market_dividends.parquet')
        print("M0 baseline artifacts saved.")

    # ── Train models ──
    for model_type in models_list:
        print("\n" + "=" * 60)
        print(f"Training: {model_type}")
        print("=" * 60)

        bag_seeds_list = [int(s) for s in args.bag_seeds.split(',') if s.strip()]
        use_bagging = (model_type == "Offset" and len(bag_seeds_list) > 1)

        if model_type == "Exploded Logit":
            model = ExplodedLogit(cv=args.cv_folds, backend='xgboost', params=xgb_params_dict, rounds=args.rounds, top_n=args.top_n)
            model.fit(X_train, Y_train)
            result_df = model.predict_proba(X_test)
        elif model_type == "Softmax":
            model = SoftmaxModel(cv=args.cv_folds, backend='xgboost', params=xgb_params_dict, rounds=args.rounds)
            model.fit(X_train, Y_train)
            result_df = model.predict_proba(X_test)
        elif model_type == "Offset":
            # Pass ALL user-supplied XGBoost params to OffsetModel (they override its defaults).
            # Defaults in the Streamlit UI match OffsetModel.DEFAULT_PARAMS so the user sees
            # the same values that will be used.
            if use_bagging:
                # Train one Offset per seed, average predictions, renormalize per race.
                # `model` = last-trained so imp_df / booster exist for downstream artifact saving.
                preds_list = []
                for seed in bag_seeds_list:
                    print(f"  Bagging seed={seed}...")
                    seed_params = {**xgb_params_dict, 'seed': seed}
                    m = OffsetModel(backend='xgboost', params=seed_params, rounds=args.rounds)
                    m.fit(X_train, Y_train)
                    preds_list.append(m.predict_proba(X_test)['Y_prob'])
                    model = m
                stacked = pd.concat(preds_list, axis=1)
                bagged = stacked.mean(axis=1)
                bagged = bagged / bagged.groupby(level=0).transform('sum')
                result_df = pd.DataFrame({'Y_prob': bagged})
            else:
                model = OffsetModel(backend='xgboost', params=xgb_params_dict, rounds=args.rounds)
                model.fit(X_train, Y_train)
                result_df = model.predict_proba(X_test)
        else:
            raise ValueError(f"Unknown model: {model_type}")

        # ── Isotonic calibration ──
        if args.calibrate and len(train_years_list) >= 2:
            cal_year = train_years_list[-1]
            cal_train_years = train_years_list[:-1]
            print(f"  Calibrating: training on {cal_train_years[0]}-{cal_train_years[-1]}, "
                  f"calibration on {cal_year}")

            # Re-predict on calibration year
            df_cal = df.copy()
            X_cal_train, X_cal_test, Y_cal_train, Y_cal_test = split_data(
                df_cal, cal_train_years, [cal_year], 'y.status_place',
                remove_ent=True, shuffle=False
            )
            cal_preds = model.predict_proba(X_cal_test)
            cal_win = (Y_cal_test == 1).astype(int)
            cal_p = cal_preds['Y_prob'].reindex(Y_cal_test.index).dropna()
            valid_mask = cal_win.reindex(cal_p.index).notna()

            iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            iso.fit(cal_p[valid_mask].values, cal_win.reindex(cal_p.index)[valid_mask].values)

            # Apply to test predictions
            raw_preds = result_df['Y_prob'].dropna()
            calibrated = iso.predict(raw_preds.values)
            calibrated_series = pd.Series(calibrated, index=raw_preds.index)

            # Renormalize within race
            race_sums = calibrated_series.groupby(level=0).transform('sum')
            calibrated_series = calibrated_series / race_sums

            result_df = pd.DataFrame({'Y_prob': calibrated_series})
            print(f"  Calibration applied. Pred range: {calibrated_series.min():.4f} - {calibrated_series.max():.4f}")

        ret, Y_test_win, dividends_df = _build_eval_frame(Y_test, result_df, odds)

        prefix = model_type.lower().replace(" ", "_")
        if args.calibrate:
            prefix += "_cal"
        _run_evaluation(ret, args.alpha, initial_bankroll, prefix)

        # Save artifacts
        result_df.to_parquet(f'results/{prefix}_result_df.parquet')
        Y_test_win.to_frame().to_parquet(f'results/{prefix}_y_test_win.parquet')
        odds.loc[Y_test.index].reset_index().to_parquet(f'results/{prefix}_odds.parquet')
        dividends_df.to_parquet(f'results/{prefix}_dividends.parquet')

        if model.imp_df is not None:
            model.imp_df.to_parquet(f'results/{prefix}_feature_importance.parquet')

        print(f"{model_type} completed.")


if __name__ == "__main__":
    main()
