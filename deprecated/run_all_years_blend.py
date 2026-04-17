"""Run Softmax model for each test year with market blending at various weights."""
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

blend_weights = [1.0, 0.75, 0.50, 0.25]  # model weight
all_results = {w: [] for w in blend_weights}

for test_year in range(2015, 2025):
    train_years = list(range(2010, test_year))
    print(f"\n{'='*60}")
    print(f"Test year: {test_year}")
    print(f"{'='*60}")

    try:
        df_copy = df.drop(columns=['Win Odds'])
        X_train, X_test, Y_train, Y_test = split_data(
            df_copy, train_years, [test_year], 'y.status_place',
            remove_ent=True, shuffle=False
        )

        model = SoftmaxModel(cv=3, backend='xgboost', params=xgb_params, rounds=200)
        model.fit(X_train, Y_train)
        result_df = model.predict_proba(X_test)

        Y_test_win = (Y_test == 1).astype(int).rename('y.status_win')
        odds = odds_all.reindex(Y_test.index).dropna()

        # Build eval frame
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

        # Market prob aligned to ret
        ret_idx = ret.set_index(list(Y_test.index.names))
        market_prob = (1.0 / odds.loc[ret_idx.index])
        market_prob = market_prob / market_prob.groupby(level=0).transform('sum')
        market_vals = market_prob.values

        # Test each blend weight
        for w in blend_weights:
            blended = w * ret['pred_prob'].values + (1 - w) * market_vals
            # Renormalize within race
            ret_tmp = ret.copy()
            ret_tmp['pred_prob'] = blended

            ll = sk_log_loss(ret_tmp['y.status_win'], ret_tmp['pred_prob'], labels=[0, 1])

            race_df, bet_df, bw, br = simulate_bankroll(
                ret_tmp, alpha, initial_bankroll=initial_bankroll,
                probability_col='pred_prob', kelly_odds_col='Win Odds',
                payoff_col='Returns', result_col='y.status_win',
            )
            final_br = br.iloc[-1] if len(br) > 0 else initial_bankroll
            roi = (final_br - initial_bankroll) / initial_bankroll
            n_bets = len(bet_df) if bet_df is not None else 0

            all_results[w].append({
                'year': test_year, 'll': ll, 'final_bankroll': final_br,
                'roi': roi, 'n_bets': n_bets,
            })

        # Print this year
        ll_mkt = sk_log_loss(ret['y.status_win'], market_vals, labels=[0, 1])
        print(f"  Market LL: {ll_mkt:.5f}")
        for w in blend_weights:
            r = all_results[w][-1]
            beat = '*' if r['ll'] < ll_mkt else ' '
            print(f"  Blend {w:.0%} model: LL={r['ll']:.5f}{beat} Bankroll=${r['final_bankroll']:,.0f} ROI={r['roi']:+.2%}")

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Summary per blend weight
print(f"\n{'='*80}")
print("SUMMARY BY BLEND WEIGHT")
print(f"{'='*80}")

for w in blend_weights:
    rdf = pd.DataFrame(all_results[w])
    pos_roi = (rdf['roi'] > 0).sum()
    avg_roi = rdf['roi'].mean()
    cum = (1 + rdf['roi']).prod() - 1
    avg_ll = rdf['ll'].mean()
    print(f"\n  Model weight {w:.0%}:")
    print(f"    Avg LL: {avg_ll:.5f}")
    print(f"    Positive ROI: {pos_roi}/{len(rdf)} years")
    print(f"    Avg ROI: {avg_roi:+.2%}")
    print(f"    Cumulative: {cum:+.2%}")
    parts = [f"{row['year']}:{row['roi']:+.1%}" for _, row in rdf.iterrows()]
    print(f"    Year-by-year: {', '.join(parts)}")
