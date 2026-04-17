"""Run all years with REGULARIZED params (the profitable config)."""
import sys, os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss as sk_log_loss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from model import split_data
from model_classifier import SoftmaxModel
from betting import simulate_bankroll
from utilities import conv_dict, load_pool_dividends

df = pd.read_parquet("../data/processed/race_features.parquet")
print(f"Loaded {len(df):,} rows, {len(df.columns)} cols")
feature_list = [c for c in df.columns if c not in ['Win Odds', 'y.status_place', 'Year']]
print(f"Features ({len(feature_list)}): {feature_list}")

odds_all = df['Win Odds']
initial_bankroll = 10_000_000.0
alpha = 0.01

dividends = conv_dict(load_pool_dividends())
dividends_df = dividends.reset_index() if dividends.index.names != [None] else dividends.copy()

# Regularized params (the profitable config)
reg_params = {'learning_rate': 0.02, 'max_depth': 4, 'min_child_weight': 20,
              'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.1,
              'reg_alpha': 1.0, 'reg_lambda': 5.0, 'tree_method': 'auto',
              'nthread': 10, 'seed': 42}

results_reg = []
results_cal = []

for test_year in range(2015, 2025):
    train_years = list(range(2010, test_year))
    print(f"\n{'='*60}")
    print(f"Test year: {test_year}")
    print(f"{'='*60}")

    df_copy = df.drop(columns=['Win Odds'])
    X_train, X_test, Y_train, Y_test = split_data(
        df_copy, train_years, [test_year], 'y.status_place',
        remove_ent=True, shuffle=False
    )
    print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

    model = SoftmaxModel(cv=3, backend='xgboost', params=reg_params, rounds=300)
    model.fit(X_train, Y_train)
    result_df = model.predict_proba(X_test)

    # Feature importance
    if model.imp_df is not None:
        fi = model.imp_df.copy()
        if 'feature' in fi.columns:
            fi = fi.set_index('feature')
        num_col = fi.select_dtypes(include='number').columns[0]
        print(f"  Top 5: {', '.join(f'{idx}={row[num_col]:.1f}' for idx, row in fi.nlargest(5, num_col).iterrows())}")

    Y_test_win = (Y_test == 1).astype(int).rename('y.status_win')
    odds = odds_all.reindex(Y_test.index).dropna()
    preds_raw = result_df['Y_prob'].reindex(Y_test.index).dropna()

    # Calibrated version
    cal_year = train_years[-1]
    df_cal = df.drop(columns=['Win Odds'])
    X_cal, _, Y_cal, _ = split_data(df_cal, train_years[:-1], [cal_year], 'y.status_place',
                                     remove_ent=True, shuffle=False)
    cal_preds = model.predict_proba(X_cal)
    cal_win = (Y_cal == 1).astype(int)
    cal_p = cal_preds['Y_prob'].reindex(Y_cal.index).dropna()
    valid_mask = cal_win.reindex(cal_p.index).notna()
    iso = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    iso.fit(cal_p[valid_mask].values, cal_win.reindex(cal_p.index)[valid_mask].values)
    preds_cal = pd.Series(iso.predict(preds_raw.values), index=preds_raw.index)
    race_sums = preds_cal.groupby(level=0).transform('sum')
    preds_cal = preds_cal / race_sums

    for config_name, preds in [('regularized', preds_raw), ('reg_calibrated', preds_cal)]:
        common = preds.index.intersection(Y_test_win.index).intersection(odds.index)
        ret = pd.DataFrame({
            'y.status_win': Y_test_win.loc[common].values,
            'pred_prob': preds.loc[common].values,
        }, index=common).reset_index()

        race_id_name = Y_test.index.names[0]
        ret = ret.merge(dividends_df, left_on=race_id_name,
                        right_on=dividends_df.columns[0] if race_id_name not in dividends_df.columns else race_id_name,
                        how='left', suffixes=('', '_div'))
        odds_flat = odds.loc[common].reset_index()
        ret = ret.merge(odds_flat, on=list(Y_test.index.names), how='left', suffixes=('', '_odds'))
        if 'race_id' not in ret.columns and race_id_name in ret.columns:
            ret = ret.rename(columns={race_id_name: 'race_id'})

        ret_idx = ret.set_index(list(Y_test.index.names))
        mkt_prob = (1.0 / odds.reindex(ret_idx.index))
        mkt_prob = mkt_prob / mkt_prob.groupby(level=0).transform('sum')

        ll_model = sk_log_loss(ret['y.status_win'], ret['pred_prob'], labels=[0, 1])
        ll_market = sk_log_loss(ret['y.status_win'], mkt_prob.values, labels=[0, 1])

        race_df, bet_df, bw, br = simulate_bankroll(
            ret, alpha, initial_bankroll=initial_bankroll,
            probability_col='pred_prob', kelly_odds_col='Win Odds',
            payoff_col='Returns', result_col='y.status_win',
        )
        final_br = br.iloc[-1] if len(br) > 0 else initial_bankroll
        roi = (final_br - initial_bankroll) / initial_bankroll

        result_list = results_reg if config_name == 'regularized' else results_cal
        result_list.append({
            'year': test_year, 'll_model': ll_model, 'll_market': ll_market,
            'roi': roi, 'bankroll': final_br,
        })
        beat = '*' if ll_model < ll_market else ' '
        print(f"  {config_name:>20}: LL={ll_model:.5f}{beat} mkt={ll_market:.5f} ROI={roi:+.2%}")

# Summary
for name, results in [('REGULARIZED', results_reg), ('REG + CALIBRATED', results_cal)]:
    rdf = pd.DataFrame(results)
    beats = (rdf['ll_model'] < rdf['ll_market']).sum()
    pos = (rdf['roi'] > 0).sum()
    cum = (1 + rdf['roi']).prod() - 1
    print(f"\n{'='*60}")
    print(f"{name}:")
    print(f"  Avg LL: {rdf['ll_model'].mean():.5f} (market: {rdf['ll_market'].mean():.5f})")
    print(f"  Beats market LL: {beats}/{len(rdf)}")
    print(f"  Positive ROI: {pos}/{len(rdf)}")
    print(f"  Avg ROI: {rdf['roi'].mean():+.2%}")
    print(f"  Cumulative: {cum:+.2%}")
    parts = [f"{int(r['year'])}:{r['roi']:+.1%}" for _, r in rdf.iterrows()]
    print(f"  Years: {', '.join(parts)}")
