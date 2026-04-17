"""Run Softmax model for each test year, report per-year Kelly results."""
import sys, os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from model import split_data
from model_classifier import SoftmaxModel
from betting import simulate_bankroll
from utilities import conv_dict, load_pool_dividends
from sklearn.metrics import log_loss as sk_log_loss

df = pd.read_parquet("../data/processed/race_features.parquet")
print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")
feature_list = [c for c in df.columns if c not in ['Win Odds', 'y.status_place', 'Year']]
print(f"Features ({len(feature_list)}): {feature_list}")

odds_all = df['Win Odds']
initial_bankroll = 10_000_000.0
alpha = 0.01
xgb_params = {'learning_rate': 0.05, 'max_depth': 6, 'min_child_weight': 1,
               'subsample': 1, 'colsample_bytree': 1, 'gamma': 0, 'reg_alpha': 0, 'tree_method': 'auto'}

dividends = conv_dict(load_pool_dividends())
dividends_df = dividends.reset_index() if dividends.index.names != [None] else dividends.copy()

results = []

for test_year in range(2015, 2025):
    train_years = list(range(2010, test_year))
    print(f"\n{'='*60}")
    print(f"Test year: {test_year}, Train: {train_years[0]}-{train_years[-1]}")
    print(f"{'='*60}")

    try:
        df_copy = df.drop(columns=['Win Odds'])
        X_train, X_test, Y_train, Y_test = split_data(
            df_copy, train_years, [test_year], 'y.status_place',
            remove_ent=True, shuffle=False
        )
        print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

        if len(X_test) == 0:
            continue

        model = SoftmaxModel(cv=3, backend='xgboost', params=xgb_params, rounds=200)
        model.fit(X_train, Y_train)
        result_df = model.predict_proba(X_test)

        # Feature importance
        if model.imp_df is not None:
            fi = model.imp_df.copy()
            if 'feature' in fi.columns:
                fi = fi.set_index('feature')
            num_col = fi.select_dtypes(include='number').columns[0]
            print(f"  Top 5: {', '.join(f'{idx}={row[num_col]:.1f}' for idx, row in fi.nlargest(5, num_col).iterrows())}")

        # Eval
        Y_test_win = (Y_test == 1).astype(int).rename('y.status_win')
        odds = odds_all.reindex(Y_test.index).dropna()

        ret = Y_test_win.reset_index()
        result_flat = result_df[['Y_prob']].reset_index()
        ret = ret.merge(result_flat, on=list(Y_test.index.names), how='inner')
        ret = ret.rename(columns={'Y_prob': 'pred_prob'})

        race_id_name = Y_test.index.names[0]
        ret = ret.merge(dividends_df, left_on=race_id_name,
                        right_on=dividends_df.columns[0] if race_id_name not in dividends_df.columns else race_id_name,
                        how='left', suffixes=('', '_div'))
        odds_flat = odds.reset_index()
        ret = ret.merge(odds_flat, on=list(Y_test.index.names), how='left', suffixes=('', '_odds'))
        if 'race_id' not in ret.columns and race_id_name in ret.columns:
            ret = ret.rename(columns={race_id_name: 'race_id'})

        # Market prob (aligned to ret)
        ret_idx = ret.set_index(list(Y_test.index.names))
        market_prob = (1.0 / odds.loc[ret_idx.index])
        market_prob = market_prob / market_prob.groupby(level=0).transform('sum')

        ll_model = sk_log_loss(ret['y.status_win'], ret['pred_prob'], labels=[0, 1])
        ll_market = sk_log_loss(ret['y.status_win'], market_prob.values, labels=[0, 1])

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
            'll_model': ll_model, 'll_market': ll_market,
            'final_bankroll': final_br, 'roi': roi,
        })
        print(f"  LL model={ll_model:.5f}, market={ll_market:.5f}")
        print(f"  Bankroll: ${final_br:,.0f} (ROI={roi:+.2%}), bets={n_bets}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary
print(f"\n{'='*80}")
print("YEAR-BY-YEAR SUMMARY")
print(f"{'='*80}")
print(f"{'Year':>6} | {'N_Test':>7} | {'N_Bets':>7} | {'LL Model':>10} | {'LL Market':>10} | {'Bankroll':>12} | {'ROI':>8}")
print("-" * 80)

for r in results:
    beat = '*' if r['ll_model'] < r['ll_market'] else ' '
    print(f"{r['year']:>6} | {r['n_test']:>7,} | {r['n_bets']:>7,} | {r['ll_model']:>10.5f}{beat}| {r['ll_market']:>10.5f} | ${r['final_bankroll']:>11,.0f} | {r['roi']:>+7.2%}")

rdf = pd.DataFrame(results)
print(f"\nAvg LL model: {rdf['ll_model'].mean():.5f}")
print(f"Avg LL market: {rdf['ll_market'].mean():.5f}")
print(f"Model beats market LL: {(rdf['ll_model'] < rdf['ll_market']).sum()}/{len(rdf)} years")
print(f"Positive ROI years: {(rdf['roi'] > 0).sum()}/{len(rdf)}")
print(f"Avg ROI: {rdf['roi'].mean():+.2%}")
cumulative = (1 + rdf['roi']).prod() - 1
print(f"Cumulative ROI (compounded): {cumulative:+.2%}")
