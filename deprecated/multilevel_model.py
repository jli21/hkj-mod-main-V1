"""
multilevel_model.py
===================
Time-respecting BLUP feature extraction with:
  - Epoch-based refitting (monthly, not daily)
  - Proper track×date random intercepts (not OLS residual medians)
  - Data-driven rider shrinkage
  - BLUP uncertainty output for downstream Kelly damping
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.mixed_linear_model import MixedLM
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# Stage 1: Track × Date variant (day-level random intercept)
# ---------------------------------------------------------------------------
def _first_stage_variant(train_df):
    """
    Estimate track×date random intercepts via a lightweight mixed model.
    Falls back to OLS median-residual if the mixed model fails.
    """
    df = train_df.copy()
    required = {
        'log_time', 'log_dist', 'RaceClass_ord', 'Track_Turf',
        'Going_ORD', 'Draw', 'Act_Wt', 'Racecourse', 'r_date',
    }
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    for c in ['Track_Turf']:
        df[c] = df[c].astype('category')

    # Group key for the random intercept
    df['track_date'] = (
        df['Racecourse'].astype(str) + '_' + df['r_date'].astype(str)
    )

    # Going as ordinal (not categorical) — saves df, respects ordering
    formula = (
        "log_time ~ log_dist + RaceClass_ord + C(Track_Turf) "
        "+ Going_ORD + Draw + Act_Wt"
    )

    try:
        model = MixedLM.from_formula(
            formula=formula,
            re_formula="1",
            groups="track_date",
            data=df,
        )
        res = model.fit(reml=True, method='lbfgs', maxiter=150, disp=False)

        # Extract random intercepts keyed by (Racecourse, r_date)
        variant_dict = {k: float(np.asarray(v).ravel()[0])
                        for k, v in res.random_effects.items()}
        variant = pd.Series(variant_dict, name='variant')
        parts = variant.index.str.rsplit('_', n=1, expand=True)
        variant.index = pd.MultiIndex.from_arrays(
            [parts.get_level_values(0).astype(int),
             pd.to_datetime(parts.get_level_values(1))],
            names=['Racecourse', 'r_date'],
        )
        return variant

    except Exception:
        # Fallback: OLS residual medians (no Racecourse in FE to avoid
        # double-counting when the second stage also includes it)
        formula_fe = (
            "log_time ~ log_dist + RaceClass_ord + C(Track_Turf) "
            "+ Going_ORD + Draw + Act_Wt"
        )
        ols = smf.ols(formula_fe, data=df).fit()
        resid = df['log_time'] - ols.fittedvalues
        variant = (
            df.assign(resid=resid)
            .groupby(['Racecourse', 'r_date'])['resid']
            .median()
            .rename('variant')
        )
        return variant


# ---------------------------------------------------------------------------
# Stage 2: Participant mixed model on variant-adjusted times
# ---------------------------------------------------------------------------
def _fit_mixedlm(train_df, variant_map, use_slope=True):
    """
    Fit MixedLM on variant-adjusted log_time with participant random effects.
    Returns (result, dist_mean).
    """
    df = train_df.copy()
    # Defensive: deduplicate variant_map index to prevent join errors
    vmap = variant_map[~variant_map.index.duplicated(keep='first')]
    df = df.join(vmap, on=['Racecourse', 'r_date'])
    df['variant'] = df['variant'].fillna(0.0)
    df['log_time_adj'] = df['log_time'] - df['variant']

    dist_mean = df['log_dist'].mean()
    df['log_dist_c'] = df['log_dist'] - dist_mean

    for c in ['Track_Turf']:
        df[c] = df[c].astype('category')

    # Going treated as ordinal continuous, not categorical
    fe = ("log_time_adj ~ log_dist + RaceClass_ord + C(Track_Turf) "
          "+ Going_ORD + Draw + Act_Wt")

    # Drop rows with NaN in any model column to avoid shape mismatch.
    # Keep the original index so fittedvalues stays aligned with the
    # caller's (race_id, Horse No.) MultiIndex.
    model_cols = ['log_time_adj', 'log_dist', 'RaceClass_ord', 'Track_Turf',
                  'Going_ORD', 'Draw', 'Act_Wt', 'Participant_link', 'log_dist_c']
    df = df.dropna(subset=model_cols)

    re_formula = "1 + log_dist_c" if use_slope else "1"

    model = MixedLM.from_formula(
        formula=fe,
        re_formula=re_formula,
        groups="Participant_link",
        data=df,
    )
    res = model.fit(reml=True, method='lbfgs', maxiter=200, disp=False)
    return res, dist_mean


# ---------------------------------------------------------------------------
# Stage 3: Rider empirical-Bayes shrinkage (data-driven k)
# ---------------------------------------------------------------------------
def _compute_rider_shrunk_resid(train_df, fitted, k=None):
    """
    Rider residual mean with empirical-Bayes shrinkage.
    If k is None, estimate it from the variance decomposition.
    """
    resid = train_df['log_time'] - fitted
    groups = train_df.assign(resid=resid).groupby('Rider')['resid']

    n = groups.size()
    mu = groups.mean()

    if k is None:
        sigma2_within = groups.var().mean()
        if np.isnan(sigma2_within) or sigma2_within == 0:
            sigma2_within = 1e-6
        sigma2_between = max(mu.var() - sigma2_within / max(n.mean(), 1), 1e-8)
        k = sigma2_within / sigma2_between

    shrunk = (n / (n + k)) * mu
    return shrunk.to_dict()


# ---------------------------------------------------------------------------
# Apply cached model to a single date's runners
# ---------------------------------------------------------------------------
def _apply_blups_to_date(df, current_idx, res, dist_mean,
                         rider_map, participant_col, rider_col):
    """
    Vectorised application of cached model results to one race-date.
    Writes: participant_re_intercept, participant_re_slope,
            PPM_entry, PPM_uncertainty, rider_resid_mean_shrunk.
    """
    dist_c = df.loc[current_idx, 'log_dist'].values - dist_mean
    pids = df.loc[current_idx, participant_col].values
    riders = df.loc[current_idx, rider_col].astype(str).values

    re = res.random_effects
    re_cov = res.cov_re  # covariance matrix of random effects

    n = len(current_idx)
    re_int = np.zeros(n)
    re_slo = np.zeros(n)
    blup_var = np.zeros(n)

    has_slope = (re_cov.shape[0] > 1)

    # Pre-extract covariance elements
    var0 = float(re_cov.iloc[0, 0])
    var1 = float(re_cov.iloc[1, 1]) if has_slope else 0.0
    cov01 = float(re_cov.iloc[0, 1]) if has_slope else 0.0

    for j, pid in enumerate(pids):
        dc = dist_c[j]
        if pid in re:
            arr = np.asarray(re[pid]).ravel()
            re_int[j] = arr[0]
            re_slo[j] = arr[1] if len(arr) > 1 else 0.0
        # else: stays 0.0 (population mean)

        # Posterior variance of linear predictor a0 + a1*dc
        if has_slope:
            blup_var[j] = var0 + dc ** 2 * var1 + 2 * dc * cov01
        else:
            blup_var[j] = var0

    contrib = re_int + re_slo * dist_c

    df.loc[current_idx, 'participant_re_intercept'] = re_int
    df.loc[current_idx, 'participant_re_slope'] = re_slo
    df.loc[current_idx, 'PPM_entry'] = -contrib  # higher = better
    df.loc[current_idx, 'PPM_uncertainty'] = np.sqrt(np.maximum(blup_var, 0.0))
    df.loc[current_idx, 'rider_resid_mean_shrunk'] = [
        rider_map.get(r, 0.0) for r in riders
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_time_respecting_blups(
    df,
    *,
    min_train_size=100,
    refit_every_days=30,
    rolling_days=365 * 3,
    participant_col='Participant_link',
    rider_col='Rider',
    date_col='r_date',
    track_col='Racecourse',
):
    """
    Time-respecting BLUP features with epoch-based refitting.

    Parameters
    ----------
    df : DataFrame
        Must contain: log_time, log_dist, RaceClass_ord, Racecourse,
        Track_Turf, Going_ORD, Draw, Act_Wt, r_date, Participant_link, Rider.
    min_train_size : int
        Minimum rows in the rolling window before fitting.
    refit_every_days : int
        How often (in calendar days) to refit the mixed model.
    rolling_days : int
        Width of the rolling training window.

    Returns
    -------
    DataFrame with added columns:
        participant_re_intercept, participant_re_slope,
        rider_resid_mean_shrunk, PPM_entry, PPM_uncertainty.
    """
    df = df.copy()

    if 'log_time' not in df.columns:
        df['log_time'] = np.log(df['time_sec'])
    if 'log_dist' not in df.columns:
        raise ValueError("Column 'log_dist' is required.")

    # Safety net: no-op if prep_data already deduped, but guards against
    # callers who bypass prep_data.
    n_before = len(df)
    df = df[~df.index.duplicated(keep='first')].copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  Dropped {n_dropped} duplicate index entries")

    df[date_col] = pd.to_datetime(df[date_col])
    all_dates = np.sort(df[date_col].unique())

    # Initialise output columns
    out_cols = [
        'participant_re_intercept', 'participant_re_slope',
        'rider_resid_mean_shrunk', 'PPM_entry', 'PPM_uncertainty',
    ]
    for col in out_cols:
        df[col] = np.nan

    # Cached model state
    cached_res = None
    cached_variant = None
    cached_dist_mean = None
    cached_rider_map = None
    last_refit = None

    n_refits = 0
    n_dates_applied = 0

    for current_date in all_dates:
        start_date = current_date - pd.Timedelta(days=rolling_days)
        train = df[(df[date_col] < current_date) & (df[date_col] >= start_date)].copy()
        current_mask = df[date_col] == current_date
        current_idx = df.index[current_mask]

        if len(train) < min_train_size or len(current_idx) == 0:
            continue

        # ---- Decide whether to refit ----
        needs_refit = (
            cached_res is None
            or last_refit is None
            or (pd.Timestamp(current_date) - pd.Timestamp(last_refit)).days >= refit_every_days
        )

        if needs_refit:
            try:
                cached_variant = _first_stage_variant(train)

                try:
                    cached_res, cached_dist_mean = _fit_mixedlm(
                        train, cached_variant, use_slope=True
                    )
                except Exception:
                    cached_res, cached_dist_mean = _fit_mixedlm(
                        train, cached_variant, use_slope=False
                    )

                # Rider residuals
                # Defensive: deduplicate variant index to prevent join errors
                cv_dedup = cached_variant[~cached_variant.index.duplicated(keep='first')]
                train_v = train.join(cv_dedup, on=[track_col, date_col])
                train_v['variant'] = train_v['variant'].fillna(0.0)
                train_v['log_time_adj'] = train_v['log_time'] - train_v['variant']
                # fittedvalues may be shorter (NaN rows dropped during fit);
                # align on intersection and fill missing with 0
                fitted_adj = pd.Series(cached_res.fittedvalues)
                fitted_full = fitted_adj.reindex(train_v.index, fill_value=0.0) + train_v['variant']
                cached_rider_map = _compute_rider_shrunk_resid(
                    train_v, fitted_full, k=None  # data-driven
                )

                last_refit = current_date
                n_refits += 1

            except Exception as e:
                print(f"[{pd.Timestamp(current_date).date()}] Refit failed: {e}")
                # If we have no cached model at all, skip this date
                if cached_res is None:
                    continue
                # Otherwise fall through and apply the stale cached model

        # ---- Apply cached model to today's runners ----
        try:
            _apply_blups_to_date(
                df, current_idx, cached_res, cached_dist_mean,
                cached_rider_map, participant_col, rider_col,
            )
            n_dates_applied += 1
        except Exception as e:
            print(f"[{pd.Timestamp(current_date).date()}] Apply failed: {e}")
            continue

    # Impute remaining NaNs to population mean (0)
    for col in out_cols:
        df[col] = df[col].fillna(0.0)

    df.sort_index(inplace=True)
    print(f"Done: {n_refits} model refits applied across {n_dates_applied} race dates.")
    return df
