"""
betting.py
==========
Kelly criterion bet sizing with:
  - Analytical gradients for robust optimisation
  - Edge filtering
  - BLUP uncertainty damping
  - Bankroll simulation with rebate logic
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from utilities import *

import warnings
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------
MIN_BET = 10.0


def multinomial_kelly(alpha, p, b, max_per_horse=0.15, max_total=0.30, eps=1e-12):
    """
    Multinomial Kelly with analytical gradient and log-barrier.

    Parameters
    ----------
    alpha : float
        Fractional Kelly scaling (0 < alpha <= 1 recommended).
    p : np.ndarray
        Estimated win probabilities for each horse in the race.
    b : np.ndarray
        Net odds for each horse (decimal_odds - 1).
    max_per_horse : float
        Maximum fraction of bankroll on any single horse.
    max_total : float
        Maximum total fraction of bankroll wagered per race.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Optimal (scaled) bet fractions.
    """
    n = len(p)

    def neg_expected_log_growth(f):
        F = np.sum(f)
        R = 1.0 - F + (1.0 + b) * f
        R = np.maximum(R, eps)
        return -np.dot(p, np.log(R))

    def gradient(f):
        F = np.sum(f)
        R = 1.0 - F + (1.0 + b) * f
        R = np.maximum(R, eps)
        p_over_R = p / R
        common = -p_over_R.sum()          # from the -1 (total bet) term
        grad = common + p_over_R * (1.0 + b)
        return -grad

    x0 = np.zeros(n)
    bounds = [(0.0, max_per_horse) for _ in range(n)]
    constraints = {'type': 'ineq', 'fun': lambda f: max_total - np.sum(f)}

    result = minimize(
        neg_expected_log_growth, x0,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method='SLSQP',
        options={'ftol': 1e-12, 'maxiter': 500},
    )

    return alpha * np.maximum(result.x, 0.0)


def round_stakes(raw_stakes, min_bet=MIN_BET):
    """Round to nearest min_bet, zeroing out sub-threshold bets."""
    rounded = np.round(raw_stakes / min_bet) * min_bet
    rounded = np.where(raw_stakes < min_bet / 2.0, 0.0, rounded)
    return rounded


# ---------------------------------------------------------------------------
# Bet computation with edge filter + uncertainty damping
# ---------------------------------------------------------------------------
def compute_kelly_bets(
    df,
    alpha,
    probability_col='pred_prob',
    odds_col='Win Odds',
    min_edge=0.0,
    uncertainty_col=None,
):
    """
    Compute Kelly bet fractions per race, with optional edge filter
    and BLUP-uncertainty damping.

    Parameters
    ----------
    df : DataFrame
        Must have race_id in the index (level 0) or as a column.
    alpha : float
        Fractional Kelly scalar.
    probability_col : str
        Column with estimated win probabilities.
    odds_col : str
        Column with decimal odds (used for Kelly sizing).
    min_edge : float
        Minimum edge (EV - 1) to place a bet. Horses below are zeroed.
    uncertainty_col : str or None
        If provided, column with PPM_uncertainty.  Higher uncertainty
        => smaller bet (exponential damping).
    """
    results = []

    # Ensure race_id is accessible
    if 'race_id' not in df.columns:
        df_work = df.reset_index()
    else:
        df_work = df.copy()

    for race_id, group in df_work.groupby('race_id'):
        group = group.copy()
        p = group[probability_col].to_numpy().astype(float)
        b = group[odds_col].to_numpy().astype(float) - 1.0

        # Edge filter
        edge = p * (1.0 + b) - 1.0
        mask = edge > min_edge

        fractions = np.zeros(len(group))
        if mask.any():
            try:
                kelly_f = multinomial_kelly(alpha, p[mask], b[mask])
                fractions[mask] = kelly_f
            except Exception as e:
                print(f"Kelly failed for {race_id}: {e}")

        # Uncertainty damping
        if uncertainty_col and uncertainty_col in group.columns:
            u = group[uncertainty_col].to_numpy().astype(float)
            u = np.nan_to_num(u, nan=0.0)
            median_u = np.median(u[u > 0]) if (u > 0).any() else 1.0
            damping = np.exp(-u / max(median_u, 1e-8))
            fractions *= damping

        group['wagered_fraction'] = fractions
        group['estimated_edge'] = edge
        results.append(group)

    return pd.concat(results, ignore_index=False)


# ---------------------------------------------------------------------------
# Simulation: race-level returns
# ---------------------------------------------------------------------------
def simulate_kelly_bets(
    df,
    alpha,
    probability_col='pred_prob',
    odds_col='Returns',
    result_col='y.status_win',
):
    """
    Lightweight race-level return simulation (no bankroll tracking).
    """
    df_bets = compute_kelly_bets(df, alpha, probability_col, odds_col)

    if 'race_id' not in df_bets.columns:
        df_bets = df_bets.reset_index()

    race_results = []
    for race_id, group in df_bets.groupby('race_id'):
        F = group['wagered_fraction'].sum()
        winner = group[group[result_col] == 1]
        if len(winner) == 0:
            race_return = -F
        else:
            win_frac = winner.iloc[0]['wagered_fraction']
            win_odds = winner.iloc[0][odds_col]
            race_return = (1 + win_odds - 1) * win_frac + (1 - F) - 1
        race_results.append({
            'race_id': race_id,
            'total_bet_fraction': F,
            'race_return': race_return,
        })

    return pd.DataFrame(race_results)


# ---------------------------------------------------------------------------
# Full bankroll simulation with rebate
# ---------------------------------------------------------------------------
def simulate_bankroll(
    df,
    alpha,
    initial_bankroll=1_000_000.0,
    rebate_rate=0.1,
    rebate_threshold=10_000.0,
    *,
    probability_col='pred_prob',
    kelly_odds_col='Win Odds',
    payoff_col='Returns',
    result_col='y.status_win',
    uncertainty_col=None,
    min_edge=0.0,
):
    """
    Simulate a bankroll through a sequence of races.

    Parameters
    ----------
    uncertainty_col : str or None
        Pass 'PPM_uncertainty' to enable BLUP-based damping.
    min_edge : float
        Minimum edge threshold for placing a bet.

    Returns
    -------
    race_df : DataFrame
        Race-level results.
    df_bets : DataFrame
        Row-level details with bet sizes.
    bankroll_without_series : Series
        Bankroll trajectory without rebate.
    bankroll_with_series : Series
        Bankroll trajectory with rebate.
    """
    df_bets = compute_kelly_bets(
        df, alpha,
        probability_col=probability_col,
        odds_col=kelly_odds_col,
        min_edge=min_edge,
        uncertainty_col=uncertainty_col,
    )
    df_bets[payoff_col] = df_bets[payoff_col].fillna(0.0)

    # Normalised implied prob for reference
    df_bets['Implied_Prob_R'] = 1.0 / df_bets[kelly_odds_col]
    df_bets['Implied_Prob'] = (
        df_bets.groupby('race_id')['Implied_Prob_R']
        .transform(lambda x: x / x.sum())
    )
    df_bets.drop(columns=['Implied_Prob_R'], inplace=True)

    race_records = []
    bankroll_without = initial_bankroll
    bankroll_with = initial_bankroll
    bw_list = [initial_bankroll]
    bwr_list = [initial_bankroll]

    for race_id, g in df_bets.groupby('race_id', sort=False):
        # ---------- without rebate ----------
        raw_stakes_wo = g['wagered_fraction'].values * bankroll_without
        stakes_wo = round_stakes(raw_stakes_wo)
        total_bet_wo = stakes_wo.sum()
        payoff_wo = (stakes_wo * g[payoff_col].values * g[result_col].values).sum()
        bankroll_without = bankroll_without - total_bet_wo + payoff_wo
        F_wo = total_bet_wo / max(bw_list[-1], 1e-8)
        ret_wo = (bankroll_without / max(bw_list[-1], 1e-8)) - 1.0

        # ---------- with rebate ----------
        prev_bwr = bankroll_with
        raw_stakes_wr = g['wagered_fraction'].values * prev_bwr
        stakes_wr = round_stakes(raw_stakes_wr)
        total_bet_wr = stakes_wr.sum()
        loss_amount = (stakes_wr * (1 - g[result_col].values)).sum()
        rebate = rebate_rate * loss_amount if loss_amount >= rebate_threshold else 0.0
        payoff_wr = (stakes_wr * g[payoff_col].values * g[result_col].values).sum()
        bankroll_with = prev_bwr - total_bet_wr + payoff_wr + rebate
        ret_wr = (bankroll_with / max(prev_bwr, 1e-8)) - 1.0

        # Store bet sizes on the detailed frame
        g_copy = g.copy()
        g_copy['bet_size ($)'] = stakes_wr

        race_records.append({
            'race_id': race_id,
            'total_wagered_fraction': g['wagered_fraction'].sum(),
            'final_multiplier': 1.0 + ret_wo,
            'final_multiplier (rebate)': 1.0 + ret_wr,
            'race_return': ret_wo,
            'race_return (rebate)': ret_wr,
            'loss (rebate_eligibility)': loss_amount,
            'rebate_amount': rebate,
        })

        bw_list.append(bankroll_without)
        bwr_list.append(bankroll_with)

    race_df = pd.DataFrame(race_records)
    bankroll_without_series = pd.Series(bw_list[1:], index=race_df['race_id'])
    bankroll_with_series = pd.Series(bwr_list[1:], index=race_df['race_id'])

    return race_df, df_bets, bankroll_without_series, bankroll_with_series
