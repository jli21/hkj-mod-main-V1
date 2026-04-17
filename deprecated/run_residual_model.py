"""Residual model: XGBoost predicts (win - implied_prob) from edge features only.
Then corrections are added back to market probs for Kelly betting."""
import sys, os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss as sk_log_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from model import split_data
from betting import simulate_bankroll
from utilities import conv_dict, load_pool_dividends

df = pd.read_parquet("../data/processed/race_features.parquet")
print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")

# Reconstruct Implied_Prob from log_implied_prob
df['Implied_Prob'] = np.exp(df['log_implied_prob'])
df['Implied_Prob'] = df['Implied_Prob'] / df.groupby(level=0)['Implied_Prob'].transform('sum')

odds_all = df['Win Odds']
initial_bankroll = 10_000_000.0
alpha = 0.01

dividends = conv_dict(load_pool_dividends())
dividends_df = dividends.reset_index() if dividends.index.names != [None] else dividends.copy()

# Edge features ONLY — no market features
edge_features = [
    'class_change', 'class_drop_odds_stable',
    'weight_change', 'wt_z',
    'odds_change', 'recent_form', 'running_style',
    'trail_acceleration', 'trainer_track_spec', 'going_change',
    'draw_outside_ST', 'draw_inside_HV',
    # Also include non-market race features
    'Racecourse', 'Track_Turf', 'Going_ORD', 'Draw', 'Act_Wt',
]
edge_features = [f for f in edge_features if f in df.columns]
print(f"Edge features ({len(edge_features)}): {edge_features}")

results = []

for test_year in range(2015, 2025):
    train_years = list(range(2010, test_year))
    print(f"\n{'='*60}")
    print(f"Test year: {test_year}")
    print(f"{'='*60}")

    try:
        # Split data
        df_split = df.drop(columns=['Win Odds'])
        X_train_full, X_test_full, Y_train, Y_test = split_data(
            df_split, train_years, [test_year], 'y.status_place',
            remove_ent=True, shuffle=False
        )

        # Target: win indicator - implied_prob (the residual)
        train_win = (Y_train == 1).astype(float)
        test_win = (Y_test == 1).astype(float)
        train_impl = df['Implied_Prob'].loc[Y_train.index]
        test_impl = df['Implied_Prob'].loc[Y_test.index]

        target_train = train_win - train_impl  # positive = horse won and market underpriced

        # Edge features only
        X_train = X_train_full[edge_features].copy()
        X_test = X_test_full[edge_features].copy()

        print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
        print(f"  Target (residual) stats: mean={target_train.mean():.6f}, std={target_train.std():.4f}")

        # XGBoost regression on residuals — constrained
        params = {
            'objective': 'reg:squarederror',
            'learning_rate': 0.02,
            'max_depth': 3,        # shallow — prevent overfitting
            'min_child_weight': 50, # large — only find strong patterns
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 1.0,       # L1 regularization
            'reg_lambda': 5.0,      # L2 regularization
            'tree_method': 'auto',
        }

        dtrain = xgb.DMatrix(X_train, label=target_train)
        dtest = xgb.DMatrix(X_test)

        model = xgb.train(params, dtrain, num_boost_round=100,
                          evals=[(dtrain, 'train')], verbose_eval=50)

        # Predict residuals on test
        pred_residual = model.predict(dtest)
        print(f"  Predicted residual: mean={pred_residual.mean():.6f}, std={pred_residual.std():.6f}")

        # Corrected probability = market + correction
        corrected_prob = test_impl.values + pred_residual

        # Clip and renormalize within race
        corrected_prob = np.clip(corrected_prob, 0.001, 0.999)
        corrected_series = pd.Series(corrected_prob, index=Y_test.index)
        race_sums = corrected_series.groupby(level=0).transform('sum')
        corrected_prob = (corrected_series / race_sums).values

        # Log loss
        ll_corrected = sk_log_loss(test_win, corrected_prob, labels=[0, 1])
        ll_market = sk_log_loss(test_win, test_impl.values, labels=[0, 1])
        beat = '*' if ll_corrected < ll_market else ' '
        print(f"  LL corrected={ll_corrected:.5f}{beat}, market={ll_market:.5f}")

        # Build eval frame for Kelly
        odds = odds_all.reindex(Y_test.index).dropna()
        Y_test_win = test_win.rename('y.status_win')

        ret = Y_test_win.reset_index()
        ret['pred_prob'] = corrected_prob
        race_id_name = Y_test.index.names[0]
        ret = ret.merge(dividends_df, left_on=race_id_name,
                        right_on=dividends_df.columns[0] if race_id_name not in dividends_df.columns else race_id_name,
                        how='left', suffixes=('', '_div'))
        odds_flat = odds.reset_index()
        ret = ret.merge(odds_flat, on=list(Y_test.index.names), how='left', suffixes=('', '_odds'))
        if 'race_id' not in ret.columns and race_id_name in ret.columns:
            ret = ret.rename(columns={race_id_name: 'race_id'})

        # Kelly simulation
        race_df, bet_df, bw, br = simulate_bankroll(
            ret, alpha, initial_bankroll=initial_bankroll,
            probability_col='pred_prob', kelly_odds_col='Win Odds',
            payoff_col='Returns', result_col='y.status_win',
        )
        final_br = br.iloc[-1] if len(br) > 0 else initial_bankroll
        roi = (final_br - initial_bankroll) / initial_bankroll
        n_bets = len(bet_df) if bet_df is not None else 0

        results.append({
            'year': test_year, 'n_test': len(X_test), 'n_bets': n_bets,
            'll_corrected': ll_corrected, 'll_market': ll_market,
            'final_bankroll': final_br, 'roi': roi,
            'pred_resid_std': pred_residual.std(),
        })
        print(f"  Bankroll: ${final_br:,.0f} (ROI={roi:+.2%}), bets={n_bets}")

        # Feature importance
        fi = model.get_score(importance_type='gain')
        top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
        print(f"  Top 5: {', '.join(f'{k}={v:.1f}' for k,v in top5)}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print("RESIDUAL MODEL — YEAR-BY-YEAR SUMMARY")
print(f"{'='*80}")
print(f"{'Year':>6} | {'LL Correct':>10} | {'LL Market':>10} | {'Bankroll':>12} | {'ROI':>8} | {'Resid Std':>10}")
print("-" * 75)

for r in results:
    beat = '*' if r['ll_corrected'] < r['ll_market'] else ' '
    print(f"{r['year']:>6} | {r['ll_corrected']:>10.5f}{beat}| {r['ll_market']:>10.5f} | ${r['final_bankroll']:>11,.0f} | {r['roi']:>+7.2%} | {r['pred_resid_std']:>10.6f}")

rdf = pd.DataFrame(results)
print(f"\nAvg LL corrected: {rdf['ll_corrected'].mean():.5f}")
print(f"Avg LL market: {rdf['ll_market'].mean():.5f}")
print(f"Model beats market LL: {(rdf['ll_corrected'] < rdf['ll_market']).sum()}/{len(rdf)} years")
print(f"Positive ROI years: {(rdf['roi'] > 0).sum()}/{len(rdf)}")
print(f"Avg ROI: {rdf['roi'].mean():+.2%}")
cumulative = (1 + rdf['roi']).prod() - 1
print(f"Cumulative ROI (compounded): {cumulative:+.2%}")
