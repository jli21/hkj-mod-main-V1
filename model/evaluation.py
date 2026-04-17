"""
evaluation.py
=============
Model evaluation diagnostics:
  - Shrinkage-to-market sweep
  - Edge bucket realization analysis
  - Comprehensive probability / betting / bankroll metrics
"""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.linear_model import LogisticRegression

from betting import simulate_bankroll


# ---------------------------------------------------------------------------
# Shrinkage-to-market sweep
# ---------------------------------------------------------------------------
def shrinkage_sweep(
    ret,
    alpha,
    initial_bankroll=1_000_000.0,
    probability_col='pred_prob',
    kelly_odds_col='Win Odds',
    payoff_col='Returns',
    result_col='y.status_win',
    weights=(1.0, 0.75, 0.50, 0.25, 0.0),
):
    """
    Blend model probs with market probs at various weights and simulate.

    Parameters
    ----------
    ret : DataFrame
        Must have race_id, pred_prob, Win Odds, Returns, y.status_win columns.
    weights : tuple of float
        Model weights. Market weight = 1 - w.

    Returns
    -------
    DataFrame indexed by model_weight with bankroll / metric columns.
    """
    # Compute normalized market probs
    market_raw = 1.0 / ret[kelly_odds_col]
    market_prob = market_raw / ret.groupby('race_id')[kelly_odds_col].transform(
        lambda x: (1.0 / x).sum()
    )

    records = []
    for w in weights:
        blended = w * ret[probability_col] + (1.0 - w) * market_prob
        # Renormalize within race
        blended = blended / blended.groupby(ret['race_id']).transform('sum')

        ret_copy = ret.copy()
        ret_copy['pred_prob'] = blended

        try:
            race_df, bet_df, bw, br = simulate_bankroll(
                ret_copy, alpha, initial_bankroll=initial_bankroll,
                probability_col='pred_prob', kelly_odds_col=kelly_odds_col,
                payoff_col=payoff_col, result_col=result_col,
            )

            final_bw = bw.iloc[-1] if len(bw) > 0 else initial_bankroll
            final_br = br.iloc[-1] if len(br) > 0 else initial_bankroll
            n_bets = (bet_df['wagered_fraction'] > 0).sum() if 'wagered_fraction' in bet_df.columns else 0

            # Log loss and Brier
            valid = ret_copy.dropna(subset=['pred_prob', result_col])
            ll = log_loss(valid[result_col], valid['pred_prob'], labels=[0, 1])
            bs = brier_score_loss(valid[result_col], valid['pred_prob'])

            records.append({
                'model_weight': w,
                'market_weight': 1.0 - w,
                'final_bankroll_no_rebate': final_bw,
                'final_bankroll_with_rebate': final_br,
                'roi_no_rebate': (final_bw / initial_bankroll - 1.0),
                'roi_with_rebate': (final_br / initial_bankroll - 1.0),
                'log_loss': ll,
                'brier': bs,
                'n_bets': n_bets,
            })
        except Exception as e:
            print(f"Shrinkage sweep failed for w={w}: {e}")
            records.append({
                'model_weight': w,
                'market_weight': 1.0 - w,
                'final_bankroll_no_rebate': initial_bankroll,
                'final_bankroll_with_rebate': initial_bankroll,
                'roi_no_rebate': 0.0,
                'roi_with_rebate': 0.0,
                'log_loss': None,
                'brier': None,
                'n_bets': 0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Edge bucket realization
# ---------------------------------------------------------------------------
def edge_bucket_analysis(
    bet_df,
    result_col='y.status_win',
    payoff_col='Returns',
    odds_col='Win Odds',
    buckets=((0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, float('inf'))),
):
    """
    Bucket bets by estimated edge and compute realized ROI per bucket.

    Parameters
    ----------
    bet_df : DataFrame
        Output from simulate_bankroll, must have estimated_edge, wagered_fraction.

    Returns
    -------
    DataFrame with per-bucket statistics.
    """
    if 'estimated_edge' not in bet_df.columns or 'wagered_fraction' not in bet_df.columns:
        return pd.DataFrame()

    bets = bet_df[bet_df['wagered_fraction'] > 0].copy()
    if len(bets) == 0:
        return pd.DataFrame()

    records = []
    for lo, hi in buckets:
        mask = (bets['estimated_edge'] >= lo) & (bets['estimated_edge'] < hi)
        subset = bets[mask]
        if len(subset) == 0:
            records.append({
                'bucket': f"{lo*100:.1f}-{hi*100:.1f}%",
                'n_bets': 0,
                'avg_edge': None,
                'realized_roi': None,
                'win_rate': None,
                'avg_odds': None,
            })
            continue

        stakes = subset['wagered_fraction'].values
        payoffs = subset[payoff_col].fillna(0).values
        wins = subset[result_col].values
        total_staked = stakes.sum()
        total_return = (stakes * payoffs * wins).sum()
        realized_roi = (total_return / total_staked - 1.0) if total_staked > 0 else 0.0

        label = f"{lo*100:.1f}%-{hi*100:.1f}%" if hi < float('inf') else f"{lo*100:.1f}%+"
        records.append({
            'bucket': label,
            'n_bets': len(subset),
            'avg_edge': subset['estimated_edge'].mean(),
            'realized_roi': realized_roi,
            'win_rate': wins.mean(),
            'avg_odds': subset[odds_col].mean() if odds_col in subset.columns else None,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Comprehensive metrics
# ---------------------------------------------------------------------------
def comprehensive_metrics(
    pred_prob,
    y_true,
    odds,
    race_df,
    bet_df,
    bankroll_without,
    bankroll_with,
    initial_bankroll,
):
    """
    Compute probability, betting, and bankroll metrics.

    Parameters
    ----------
    pred_prob : Series or array
        Model predicted win probabilities.
    y_true : Series or array
        Binary win indicators.
    odds : Series or array
        Win odds.
    race_df : DataFrame
        Race-level results from simulate_bankroll.
    bet_df : DataFrame
        Row-level bet details from simulate_bankroll.
    bankroll_without, bankroll_with : Series
        Bankroll trajectories.
    initial_bankroll : float

    Returns
    -------
    dict with keys 'probability', 'betting', 'bankroll'.
    """
    pred_prob = np.asarray(pred_prob, dtype=float)
    y_true = np.asarray(y_true, dtype=float)

    # --- Probability metrics ---
    prob_metrics = {}
    try:
        prob_metrics['log_loss'] = float(log_loss(y_true, pred_prob, labels=[0, 1]))
    except Exception:
        prob_metrics['log_loss'] = None

    try:
        prob_metrics['brier'] = float(brier_score_loss(y_true, pred_prob))
    except Exception:
        prob_metrics['brier'] = None

    # Calibration slope
    try:
        logit_p = np.log(np.clip(pred_prob, 1e-8, 1 - 1e-8) / (1 - np.clip(pred_prob, 1e-8, 1 - 1e-8)))
        lr = LogisticRegression(fit_intercept=True, solver='lbfgs', max_iter=500)
        lr.fit(logit_p.reshape(-1, 1), y_true)
        prob_metrics['calibration_slope'] = float(lr.coef_[0][0])
        prob_metrics['calibration_intercept'] = float(lr.intercept_[0])
    except Exception:
        prob_metrics['calibration_slope'] = None
        prob_metrics['calibration_intercept'] = None

    # Top-k hit rates
    if 'race_id' in bet_df.columns and 'pred_prob' in bet_df.columns:
        for k in [1, 3]:
            hits = total = 0
            for _, group in bet_df.groupby('race_id'):
                top_idx = group['pred_prob'].nlargest(k).index
                winners = group.index[group['y.status_win'] == 1]
                if len(winners) > 0:
                    total += 1
                    if winners[0] in top_idx:
                        hits += 1
            prob_metrics[f'top{k}_hit_rate'] = hits / total if total > 0 else None
    else:
        prob_metrics['top1_hit_rate'] = None
        prob_metrics['top3_hit_rate'] = None

    # --- Betting metrics ---
    bet_metrics = {}
    if 'wagered_fraction' in bet_df.columns:
        active = bet_df[bet_df['wagered_fraction'] > 0]
        bet_metrics['n_bets'] = len(active)
        bet_metrics['turnover_fraction'] = active['wagered_fraction'].sum()
        if 'estimated_edge' in active.columns:
            bet_metrics['avg_edge'] = float(active['estimated_edge'].mean())
        else:
            bet_metrics['avg_edge'] = None
    else:
        bet_metrics['n_bets'] = 0
        bet_metrics['turnover_fraction'] = 0.0
        bet_metrics['avg_edge'] = None

    if len(race_df) > 0:
        bet_metrics['n_races'] = len(race_df)
        bet_metrics['races_with_bets'] = (race_df['total_wagered_fraction'] > 0).sum()
    else:
        bet_metrics['n_races'] = 0
        bet_metrics['races_with_bets'] = 0

    # --- Bankroll metrics ---
    bank_metrics = {}
    bw = bankroll_without.values.astype(float) if bankroll_without is not None else np.array([initial_bankroll])
    br = bankroll_with.values.astype(float) if bankroll_with is not None else np.array([initial_bankroll])

    bank_metrics['final_no_rebate'] = float(bw[-1]) if len(bw) > 0 else initial_bankroll
    bank_metrics['final_with_rebate'] = float(br[-1]) if len(br) > 0 else initial_bankroll
    bank_metrics['roi_no_rebate'] = bank_metrics['final_no_rebate'] / initial_bankroll - 1.0
    bank_metrics['roi_with_rebate'] = bank_metrics['final_with_rebate'] / initial_bankroll - 1.0

    # Average log growth per race
    if len(bw) > 1:
        bw_full = np.concatenate([[initial_bankroll], bw])
        log_returns = np.log(np.maximum(bw_full[1:], 1e-8) / np.maximum(bw_full[:-1], 1e-8))
        bank_metrics['avg_log_growth'] = float(log_returns.mean())
    else:
        bank_metrics['avg_log_growth'] = 0.0

    # Max drawdown
    if len(bw) > 0:
        running_max = np.maximum.accumulate(bw)
        drawdowns = (running_max - bw) / np.maximum(running_max, 1e-8)
        bank_metrics['max_drawdown'] = float(drawdowns.max())
    else:
        bank_metrics['max_drawdown'] = 0.0

    return {
        'probability': prob_metrics,
        'betting': bet_metrics,
        'bankroll': bank_metrics,
    }
