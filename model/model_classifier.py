# model_classifier.py
# ---------------------------------------------
# Winner-probability models with binned temperature calibration
# Supports:
#   - Exploded Logit (winner-only or top-n truncated)
#   - Softmax (winner-only)
# Includes:
#   - Leakage-safe race grouping utilities
#   - Per-race probability normalization
#   - Binned temperature scaling by field-size bins and optional surface
#   - Binary classifier helper (with calibration)
# ---------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd

# Optional: guard numba if not installed
try:
    from numba import jit
except Exception:
    def jit(nopython=True):
        def deco(f):
            return f
        return deco

from scipy.optimize import minimize
from scipy.special import softmax

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold

# Gradient boosters
import xgboost as xgb
import lightgbm as lgb


# ---------------------------
# General utilities
# ---------------------------

def normalize_probs_by_race(probs, race_index) -> np.ndarray:
    """
    Normalize predicted probabilities so they sum to 1 within each race/group.

    Parameters
    ----------
    probs : array-like
        Predicted probabilities per entry (not necessarily summing to 1 per race).
    race_index : pandas.Index (MultiIndex expected)
        Index aligned to probs. First level should be race identifier.

    Returns
    -------
    np.ndarray
        Normalized probabilities summing to 1 per race.
    """
    probs_df = pd.DataFrame({'probs': probs}, index=race_index)
    try:
        # Prefer explicit level name if available
        normalized_probs = probs_df.groupby(level='race_id').transform(lambda x: x / x.sum())
    except (KeyError, ValueError):
        # Fallback: use first level
        normalized_probs = probs_df.groupby(level=0).transform(lambda x: x / x.sum())
    return normalized_probs['probs'].values


def get_classifier(model_name, **kwargs):
    """
    Returns a scikit-learn classifier instance by name.

    Supported names:
      - 'xgboost'
      - 'lightgbm'
      - 'logistic_regression'
      - 'random_forest'
      - 'svc'
      - 'gradient_boosting'
    """
    model_name = model_name.lower()
    if model_name == 'xgboost':
        from xgboost import XGBClassifier
        clf = XGBClassifier(**kwargs)
    elif model_name == 'lightgbm':
        import lightgbm as lgbm
        clf = lgbm.LGBMClassifier(**kwargs)
    elif model_name == 'logistic_regression':
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(**kwargs)
    elif model_name == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(**kwargs)
    elif model_name == 'svc':
        from sklearn.svm import SVC
        clf = SVC(probability=True, **kwargs)
    elif model_name == 'gradient_boosting':
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(**kwargs)
    else:
        raise ValueError(
            "Model '{}' is not supported. Choose from "
            "'xgboost', 'lightgbm', 'logistic_regression', "
            "'random_forest', 'svc', 'gradient_boosting'.".format(model_name)
        )
    return clf


def binary_classifier(_clf, X_train, Y_train, X_test, Y_test, cv_value=5, normalize_probs=True):
    """
    Train a binary classifier with calibration; return calibrated probabilities on X_test.

    Notes
    -----
    - Uses CalibratedClassifierCV on a fresh clone of _clf.
    - Optionally normalizes probabilities to sum to 1 per race.

    Returns
    -------
    pandas.DataFrame
        Index aligned to X_test.index with single column 'Y_prob'.
    """
    base = clone(_clf)
    calibrated_clf = CalibratedClassifierCV(base, cv=cv_value)
    calibrated_clf.fit(X_train, Y_train)

    # Display quick feature info if available (from a separately fitted clone)
    try:
        imp_clf = clone(_clf)
        imp_clf.fit(X_train, Y_train)
        if hasattr(imp_clf, "feature_importances_"):
            scores = imp_clf.feature_importances_
        elif hasattr(imp_clf, "coef_"):
            scores = np.abs(imp_clf.coef_).ravel()
        else:
            scores = None
        if scores is not None:
            feat_imp = (
                pd.DataFrame({"Feature": X_train.columns, "Score": scores})
                .sort_values("Score", ascending=False)
                .head(50)
            )
            print("\nTop 50 features by importance/coef magnitude:\n")
            print(feat_imp.reset_index(drop=True))
        else:
            print("\n[Info] Classifier has no feature_importances_ or coef_ to display.\n")
    except Exception:
        pass

    y_train_prob = calibrated_clf.predict_proba(X_train)[:, 1]
    y_test_prob = calibrated_clf.predict_proba(X_test)[:, 1]

    if normalize_probs:
        y_train_prob = normalize_probs_by_race(y_train_prob, X_train.index)
        y_test_prob = normalize_probs_by_race(y_test_prob, X_test.index)

    y_train_bin = (y_train_prob > 0.5).astype(int)
    y_test_bin = (y_test_prob > 0.5).astype(int)

    print(f"Train Accuracy: {accuracy_score(Y_train, y_train_bin):.3f}")
    if len(Y_test) == len(y_test_bin):
        acc = accuracy_score(Y_test, y_test_bin)
    else:
        acc = accuracy_score(Y_test, y_test_bin[:len(Y_test)])
    print(f"Test Accuracy: {acc:.3f}")

    return pd.DataFrame(index=X_test.index, data=y_test_prob, columns=['Y_prob'])


def prepare_group_sizes(X: pd.DataFrame) -> np.ndarray:
    """
    Compute the number of entries per race in the order they first appear in X.

    Parameters
    ----------
    X : pandas.DataFrame
        Must be indexed by a MultiIndex whose first level is race_id.

    Returns
    -------
    np.ndarray
        Array of group sizes (field sizes), aligned to the order of races in X.
    """
    races = X.index.get_level_values(0)
    uniq_in_order = pd.Index(races).drop_duplicates()
    sizes = races.value_counts(sort=False).loc[uniq_in_order].values
    return sizes


# ---------------------------
# Temperature scaling helpers
# ---------------------------

def _group_ptr_from_sizes(sizes: np.ndarray) -> np.ndarray:
    return np.cumsum(np.concatenate([[0], sizes])).astype(np.int32)


def _race_order_and_sizes(X: pd.DataFrame):
    races = X.index.get_level_values(0)
    uniq_in_order = pd.Index(races).drop_duplicates()
    sizes = races.value_counts(sort=False).loc[uniq_in_order].values
    return uniq_in_order, sizes


def build_race_meta_from_X(
    X: pd.DataFrame,
    *,
    surface_col: str | None = None,
    distance_col: str | None = None,
    field_bins=(2, 6, 9, 99)
):
    """
    Build race-level metadata aligned to group order used by the boosters.

    Meta columns produced:
      - field_size (required)
      - surface (optional; string; 'unknown' if not provided)
      - distance_bin (optional; -1 if not provided)
      - field_bin (binned field size)

    Returns
    -------
    (race_meta: DataFrame indexed by race_id, sizes: np.ndarray)
    """
    race_ids, sizes = _race_order_and_sizes(X)
    meta = pd.DataFrame(index=race_ids)
    meta['field_size'] = sizes

    # Surface (categorical) per race
    if surface_col is not None and surface_col in X.columns:
        surface = X.groupby(level=0)[surface_col].first().reindex(race_ids)
        meta['surface'] = surface.astype(str)
    else:
        meta['surface'] = 'unknown'

    # Distance bin (optional)
    if distance_col is not None and distance_col in X.columns:
        dist = X.groupby(level=0)[distance_col].first().reindex(race_ids)
        meta['distance_bin'] = pd.cut(
            dist, bins=[0, 1200, 1600, 2000, 2600, np.inf], labels=False
        )
    else:
        meta['distance_bin'] = -1

    meta['field_bin'] = pd.cut(
        meta['field_size'], bins=[0] + list(field_bins), labels=False, right=True
    )
    return meta, sizes


def _nll_for_groups(raw: np.ndarray, true_labels: np.ndarray, group_ptr: np.ndarray) -> float:
    """
    Negative log-likelihood per race (winner-only). Supports ties by averaging winner probs.
    """
    total = 0.0
    count = 0
    for i in range(len(group_ptr) - 1):
        s, e = group_ptr[i], group_ptr[i + 1]
        y = true_labels[s:e]
        if not np.any(y == 1):
            continue
        p = softmax(raw[s:e])
        k = (y == 1).sum()
        total += -np.log(p[y == 1].sum() / max(k, 1) + 1e-12)
        count += 1
    return total / max(count, 1)


def learn_temperature_multi_global(oof_raw: np.ndarray, true_labels: np.ndarray, group_sizes: np.ndarray) -> float:
    """
    Learn a single global temperature by minimizing per-race NLL.
    """
    gp = _group_ptr_from_sizes(group_sizes)

    def obj(T):
        T = np.clip(T[0], 0.05, 10.0)
        return _nll_for_groups(oof_raw / T, true_labels, gp)

    res = minimize(obj, x0=[1.0], bounds=[(0.05, 10.0)])
    return res.x[0]


def learn_temperature_multi_binned(
    oof_raw: np.ndarray,
    true_labels: np.ndarray,
    group_sizes: np.ndarray,
    race_meta: pd.DataFrame,
    *,
    min_races_per_bin: int = 200,
    use_surface: bool = True
) -> dict:
    """
    Learn binned temperatures with fallback to a global default.

    Returns
    -------
    dict
        {'global': T_global, (field_bin, surface): T_bin, ...} or keys=(field_bin) if use_surface=False
    """
    sizes = np.asarray(group_sizes)
    gp = _group_ptr_from_sizes(sizes)

    # Global temperature
    T_global = learn_temperature_multi_global(oof_raw, true_labels, sizes)
    T_map = {'global': T_global}

    # Build per-race bin keys in order
    if use_surface and 'surface' in race_meta.columns:
        keys = list(zip(race_meta['field_bin'].astype(int).tolist(),
                        race_meta['surface'].astype(str).tolist()))
    else:
        keys = list(race_meta['field_bin'].astype(int).tolist())

    # Map bin -> race indices (by order)
    bin_to_rindices = {}
    for i in range(len(sizes)):
        key = keys[i]
        bin_to_rindices.setdefault(key, []).append(i)

    # Optimize T per bin when enough races exist
    for key, rinds in bin_to_rindices.items():
        if len(rinds) < min_races_per_bin:
            continue
        idxs = np.concatenate([np.arange(gp[i], gp[i + 1]) for i in rinds])
        sizes_bin = sizes[rinds]
        gp_bin = _group_ptr_from_sizes(sizes_bin)
        raw_bin = oof_raw[idxs]
        y_bin = true_labels[idxs]

        def obj(T):
            T = np.clip(T[0], 0.05, 10.0)
            return _nll_for_groups(raw_bin / T, y_bin, gp_bin)

        res = minimize(obj, x0=[T_global], bounds=[(0.05, 10.0)])
        T_map[key] = res.x[0]

    return T_map


def apply_temperature_binned(
    raw_preds: np.ndarray,
    group_sizes: np.ndarray,
    race_meta: pd.DataFrame,
    T_map: dict,
    *,
    use_surface: bool = True
) -> np.ndarray:
    """
    Apply binned temperature to raw logits, softmax per race, and return calibrated probabilities.
    """
    sizes = np.asarray(group_sizes)
    gp = _group_ptr_from_sizes(sizes)
    p_cal = np.empty_like(raw_preds)

    # Build per-race bin keys in order
    if use_surface and 'surface' in race_meta.columns:
        keys = list(zip(race_meta['field_bin'].astype(int).tolist(),
                        race_meta['surface'].astype(str).tolist()))
    else:
        keys = list(race_meta['field_bin'].astype(int).tolist())

    for i in range(len(sizes)):
        s, e = gp[i], gp[i + 1]
        key = keys[i]
        T = T_map.get(key, T_map['global'])
        p_cal[s:e] = softmax(raw_preds[s:e] / T)

    return p_cal


# ---------------------------
# Exploded logit (winner/top-n)
# ---------------------------

@jit(nopython=True)
def compute_exploded_grad_hess(preds, labels, group_ptr, top_n=-1):
    """
    Numba-accelerated gradients and hessians for exploded logit with optional top-n truncation.

    Args
    ----
    preds : np.ndarray
        Raw predictions (logits), shape (n_samples,)
    labels : np.ndarray
        Integer labels giving rank positions (1=winner, 2=2nd, ...), shape (n_samples,)
    group_ptr : np.ndarray
        Array of group pointers (start indices), shape (#groups+1,)
    top_n : int
        -1 = full exploded logit; 1 = softmax (winner-only); >1 = top-n truncation
    """
    n = len(preds)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)

    for i in range(len(group_ptr) - 1):
        s = group_ptr[i]
        e = group_ptr[i + 1]
        if e <= s:
            continue

        f_race = preds[s:e] - np.max(preds[s:e])
        expf = np.exp(f_race)
        pos_race = labels[s:e]

        unique_k = np.unique(pos_race)
        unique_k.sort()
        cutoff = len(unique_k) if top_n == -1 else min(top_n, len(unique_k))

        for k in unique_k[:cutoff]:
            mask = (pos_race >= k)
            if not np.any(mask):
                continue

            expf_k = expf[mask]
            Zk = np.sum(expf_k)
            if Zk == 0.0:
                continue

            pk = expf_k / Zk
            grad_stage = pk.copy()

            true_mask = (pos_race == k)
            true_idxs = np.nonzero(true_mask)[0]
            n_true = len(true_idxs)
            if n_true == 0:
                continue

            subtract = 1.0 / n_true

            mask_idxs = np.nonzero(mask)[0]
            pos_in_stage = np.searchsorted(mask_idxs, true_idxs)

            for t in range(n_true):
                idx_in_mask = pos_in_stage[t]
                grad_stage[idx_in_mask] -= subtract

            hess_stage = pk * (1.0 - pk)
            for j in range(len(mask_idxs)):
                rel_idx = mask_idxs[j]
                grad[s + rel_idx] += grad_stage[j]
                hess[s + rel_idx] += hess_stage[j]

    return grad, hess


def exploded_logit_obj(preds, data, top_n=-1):
    """
    Custom exploded logistic objective with top-n truncation for XGBoost/LightGBM.
    """
    labels = data.get_label().astype(np.int32)
    if isinstance(data, xgb.DMatrix):
        group_ptr = np.array(data.get_uint_info('group_ptr'), dtype=np.int32)
    elif isinstance(data, lgb.Dataset):
        grp = data.get_group()
        group_ptr = np.cumsum(np.concatenate(([0], grp))).astype(np.int32)
    else:
        raise ValueError("Unsupported data type for exploded_logit_obj")
    grad, hess = compute_exploded_grad_hess(preds.astype(np.float64), labels, group_ptr, top_n)
    return grad, hess


def exploded_logistic_model(X_train, X_test, Y_train, Y_test, backend, params=None, rounds=200, top_n=-1, displayFeature=True):
    """
    Train exploded-logit model and produce per-race probabilities (softmaxed within race).

    Returns
    -------
    (booster, prob_df, raw_preds, imp_df)
    """
    cat_features = X_train.select_dtypes(['category']).columns.tolist()
    grp_tr = prepare_group_sizes(X_train)
    grp_te = prepare_group_sizes(X_test)

    imp_df = None

    if backend == 'xgboost':
        xgb_params = dict(params or {})
        xgb_params.setdefault('eval_metric', 'rmse')
        xgb_params.setdefault('tree_method', 'hist')

        dtr = xgb.DMatrix(X_train, label=Y_train.astype(int), enable_categorical=True)
        dtr.set_group(grp_tr)
        dte = xgb.DMatrix(X_test, label=Y_test.astype(int), enable_categorical=True)
        dte.set_group(grp_te)

        bst = xgb.train(
            xgb_params, dtr, num_boost_round=rounds,
            obj=lambda p, d: exploded_logit_obj(p, d, top_n=top_n),
            evals=[(dtr, 'train'), (dte, 'test')],
            verbose_eval=10
        )
        raw_preds = bst.predict(dte)
        group_ptr = dte.get_uint_info('group_ptr')

        if displayFeature:
            imp_dict = bst.get_score(importance_type='gain')
            imp_df = (
                pd.DataFrame.from_dict(imp_dict, orient='index', columns=['importance'])
                .rename_axis('feature')
                .reset_index()
            )
            def resolve_feat(f):
                if f.startswith('f') and f[1:].isdigit():
                    return X_train.columns[int(f[1:])]
                return f
            imp_df['feature'] = imp_df['feature'].apply(resolve_feat)
            imp_df = imp_df.sort_values('importance', ascending=False).head(50)

    elif backend == 'lightgbm':
        lgb_params = dict(params or {})
        lgb_params.setdefault('metric', 'rmse')
        ltr = lgb.Dataset(X_train, label=Y_train.astype(int), group=grp_tr, categorical_feature=cat_features)
        lte = lgb.Dataset(X_test, label=Y_test.astype(int), group=grp_te, reference=ltr, categorical_feature=cat_features)

        bst = lgb.train(
            lgb_params, ltr, num_boost_round=rounds,
            fobj=lambda p, d: exploded_logit_obj(p, d, top_n=top_n),
            valid_sets=[ltr, lte], valid_names=['train', 'test'],
            verbose_eval=10
        )
        raw_preds = bst.predict(X_test)
        grp = lte.get_group()
        group_ptr = np.concatenate([[0], np.cumsum(grp)])

        if displayFeature:
            imp_arr = bst.feature_importance(importance_type='gain')
            imp_df = (
                pd.DataFrame({'feature': X_train.columns, 'importance': imp_arr})
                .sort_values('importance', ascending=False)
                .head(50)
            )
    else:
        raise ValueError("backend must be 'xgboost' or 'lightgbm'")

    # Softmax per race to probabilities
    p_win = np.zeros_like(raw_preds)
    for i in range(len(group_ptr) - 1):
        s, e = group_ptr[i], group_ptr[i + 1]
        f = raw_preds[s:e] - np.max(raw_preds[s:e])
        ef = np.exp(f)
        p_win[s:e] = ef / ef.sum()

    prob_df = pd.DataFrame({'Y_prob': p_win}, index=X_test.index)
    return bst, prob_df, raw_preds, imp_df


def exploded_cv(X, y, *, cv=5, backend='xgboost', params=None, rounds=200, top_n=-1):
    """
    Cross-validated out-of-fold raw logits for exploded-logit.
    Returns (oof_raw, oof_true).
    """
    groups = X.index.get_level_values(0)
    gkf = GroupKFold(n_splits=cv)
    oof_raw = np.empty(len(X))
    oof_true = np.array(y == 1, dtype=int)

    for tr_idx, val_idx in gkf.split(X, y, groups):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_val = X.iloc[val_idx]
        bst, _, raw_preds, _ = exploded_logistic_model(
            X_tr, X_val, y_tr, y.iloc[val_idx],
            backend=backend, params=params, rounds=rounds, top_n=top_n, displayFeature=False
        )
        oof_raw[val_idx] = raw_preds

    return oof_raw, oof_true


# ---------------------------
# Softmax (winner-only)
# ---------------------------

@jit(nopython=True)
def compute_softmax_grad_hess(preds, labels, group_ptr):
    """
    Numba-accelerated gradients and hessians for winner-only softmax.
    """
    n = len(preds)
    grad = np.zeros(n, dtype=np.float64)
    hess = np.zeros(n, dtype=np.float64)

    for i in range(len(group_ptr) - 1):
        s = group_ptr[i]
        e = group_ptr[i + 1]
        if e - s <= 1:
            continue

        p = preds[s:e]
        y = labels[s:e]
        winner_idxs = np.where(y == 1)[0]
        if len(winner_idxs) == 0:
            continue

        max_p = np.max(p)
        exp_p = np.exp(p - max_p)
        sum_exp = np.sum(exp_p)
        prob = np.divide(exp_p, sum_exp)

        g = prob.copy()
        subtract = 1.0 / len(winner_idxs)
        for wi in winner_idxs:
            g[wi] -= subtract

        h = prob * (1.0 - prob)

        grad[s:e] = g
        hess[s:e] = h

    return grad, hess


def softmax_obj(preds, data):
    """
    Custom softmax objective for XGBoost/LightGBM (winner-only).
    """
    labels = data.get_label()
    if isinstance(data, xgb.DMatrix):
        group_ptr = np.array(data.get_uint_info('group_ptr'), dtype=np.int32)
    elif isinstance(data, lgb.Dataset):
        grp = data.get_group()
        group_ptr = np.cumsum(np.concatenate(([0], grp))).astype(np.int32)
    else:
        raise ValueError("Unsupported data type for softmax_obj")
    grad, hess = compute_softmax_grad_hess(preds.astype(np.float64), labels.astype(np.float64), group_ptr)
    return grad, hess


def softmax_logistic_model(X_train, X_test, Y_train, Y_test, backend, params=None, rounds=200, displayFeature=True):
    """
    Train softmax model and produce per-race probabilities (softmaxed within race).

    Returns
    -------
    (booster, prob_df, raw_preds, imp_df)
    """
    cat_features = X_train.select_dtypes(['category']).columns.tolist()
    grp_tr = prepare_group_sizes(X_train)
    grp_te = prepare_group_sizes(X_test)

    Y_train_win = (Y_train == 1).astype(int)
    Y_test_win = (Y_test == 1).astype(int)

    imp_df = None

    if backend == 'xgboost':
        xgb_params = dict(params or {})
        xgb_params.setdefault('eval_metric', 'rmse')
        xgb_params.setdefault('tree_method', 'hist')

        dtr = xgb.DMatrix(X_train, label=Y_train_win, enable_categorical=True)
        dtr.set_group(grp_tr)
        dte = xgb.DMatrix(X_test, label=Y_test_win, enable_categorical=True)
        dte.set_group(grp_te)

        bst = xgb.train(
            xgb_params, dtr, num_boost_round=rounds, obj=softmax_obj,
            evals=[(dtr, 'train'), (dte, 'test')], verbose_eval=10
        )
        raw_preds = bst.predict(dte)
        group_ptr = dte.get_uint_info('group_ptr')

        if displayFeature:
            imp_dict = bst.get_score(importance_type='gain')
            imp_df = (
                pd.DataFrame.from_dict(imp_dict, orient='index', columns=['importance'])
                .rename_axis('feature')
                .reset_index()
            )
            def resolve_feat(f):
                if f.startswith('f') and f[1:].isdigit():
                    return X_train.columns[int(f[1:])]
                return f
            imp_df['feature'] = imp_df['feature'].apply(resolve_feat)
            imp_df = imp_df.sort_values('importance', ascending=False).head(50)

    elif backend == 'lightgbm':
        lgb_params = dict(params or {})
        lgb_params.setdefault('metric', 'binary')
        ltr = lgb.Dataset(X_train, label=Y_train_win, group=grp_tr, categorical_feature=cat_features)
        lte = lgb.Dataset(X_test, label=Y_test_win, group=grp_te, reference=ltr, categorical_feature=cat_features)

        bst = lgb.train(
            lgb_params, ltr, num_boost_round=rounds, fobj=softmax_obj,
            valid_sets=[ltr, lte], valid_names=['train', 'test'], verbose_eval=10
        )
        raw_preds = bst.predict(X_test)
        grp = lte.get_group()
        group_ptr = np.concatenate([[0], np.cumsum(grp)])

        if displayFeature:
            imp_arr = bst.feature_importance(importance_type='gain')
            imp_df = (
                pd.DataFrame({'feature': X_train.columns, 'importance': imp_arr})
                .sort_values('importance', ascending=False)
                .head(50)
            )
    else:
        raise ValueError("backend must be 'xgboost' or 'lightgbm'")

    # Softmax per race to probabilities
    p_win = np.zeros_like(raw_preds)
    for i in range(len(group_ptr) - 1):
        s, e = group_ptr[i], group_ptr[i + 1]
        f = raw_preds[s:e] - np.max(raw_preds[s:e])
        ef = np.exp(f)
        p_win[s:e] = ef / ef.sum()

    prob_df = pd.DataFrame({'Y_prob': p_win}, index=X_test.index)
    return bst, prob_df, raw_preds, imp_df


def softmax_cv(X, y, *, cv=5, backend='xgboost', params=None, rounds=200):
    """
    Cross-validated out-of-fold raw logits for softmax.
    Returns (oof_raw, oof_true).
    """
    y_win = (y == 1).astype(int)
    groups = X.index.get_level_values(0)
    gkf = GroupKFold(n_splits=cv)
    oof_raw = np.empty(len(X))
    oof_true = y_win

    for tr_idx, val_idx in gkf.split(X, y_win, groups):
        X_tr, y_tr = X.iloc[tr_idx], y_win[tr_idx]
        X_val = X.iloc[val_idx]
        y_val = y_win[val_idx]
        bst, _, raw_preds, _ = softmax_logistic_model(
            X_tr, X_val, y_tr, y_val,
            backend=backend, params=params, rounds=rounds, displayFeature=False
        )
        oof_raw[val_idx] = raw_preds

    return oof_raw, oof_true


# ---------------------------
# Models with binned calibration
# ---------------------------

class ExplodedLogit:
    """
    Exploded logit model with binned temperature calibration.

    Parameters
    ----------
    cv : int
        GroupKFold splits for OOF calibration logits.
    backend : {'xgboost','lightgbm'}
        Booster backend.
    params : dict
        Booster parameters.
    rounds : int
        Boosting rounds.
    top_n : int
        Exploded logit top-n truncation (-1 full, 1 winner-only, >1 top-n).
    surface_col : str or None
        Name of surface column in X to use for calibration bins (optional).
    distance_col : str or None
        Optional continuous distance feature name to build distance_bin in race_meta (not used in bins by default).
    min_races_per_bin : int
        Minimum number of races to train a bin-specific temperature; otherwise use global.
    use_surface : bool
        If True, bins are (field_bin, surface); else bins are field_bin only.
    """
    def __init__(self, cv=5, backend='xgboost', params=None, rounds=200, top_n=-1,
                 surface_col='Track_Turf', distance_col=None,
                 min_races_per_bin=200, use_surface=True):
        self.cv = cv
        self.backend = backend
        self.params = params
        self.rounds = rounds
        self.top_n = top_n
        self.surface_col = surface_col
        self.distance_col = distance_col
        self.min_races_per_bin = min_races_per_bin
        self.use_surface = use_surface

        self.T_map = None
        self.booster = None
        self.imp_df = None
        self._field_bins = (2, 6, 9, 99)  # tune as needed

    def fit(self, X, y):
        print("Starting K-fold training...")
        oof_raw, oof_true = exploded_cv(
            X, y, cv=self.cv, backend=self.backend,
            params=self.params, rounds=self.rounds, top_n=self.top_n
        )
        print("Training final iteration of model...")
        self.booster, _, _, self.imp_df = exploded_logistic_model(
            X, X.iloc[:0], y, y.iloc[:0],
            backend=self.backend, params=self.params,
            rounds=self.rounds, top_n=self.top_n, displayFeature=True
        )
        # Learn binned temperatures
        race_meta, sizes = build_race_meta_from_X(
            X, surface_col=self.surface_col, distance_col=self.distance_col, field_bins=self._field_bins
        )
        self.T_map = learn_temperature_multi_binned(
            oof_raw, oof_true, sizes, race_meta,
            min_races_per_bin=self.min_races_per_bin, use_surface=self.use_surface
        )
        return self

    def predict_proba(self, X):
        if self.backend == 'xgboost':
            dte = xgb.DMatrix(X, enable_categorical=True)
            raw = self.booster.predict(dte)
        elif self.backend == 'lightgbm':
            raw = self.booster.predict(X)
        else:
            raise ValueError("backend must be 'xgboost' or 'lightgbm'")

        race_meta, sizes = build_race_meta_from_X(
            X, surface_col=self.surface_col, distance_col=self.distance_col, field_bins=self._field_bins
        )
        p_cal = apply_temperature_binned(
            raw, sizes, race_meta, self.T_map, use_surface=self.use_surface
        )
        return pd.DataFrame(p_cal, index=X.index, columns=["Y_prob"])


class SoftmaxModel:
    """
    Softmax (winner-only) model with binned temperature calibration.

    Parameters
    ----------
    cv : int
        GroupKFold splits for OOF calibration logits.
    backend : {'xgboost','lightgbm'}
        Booster backend.
    params : dict
        Booster parameters.
    rounds : int
        Boosting rounds.
    surface_col : str or None
        Name of surface column in X to use for calibration bins (optional).
    distance_col : str or None
        Optional continuous distance feature name to build distance_bin in race_meta (not used in bins by default).
    min_races_per_bin : int
        Minimum number of races to train a bin-specific temperature; otherwise use global.
    use_surface : bool
        If True, bins are (field_bin, surface); else bins are field_bin only.
    """
    def __init__(self, cv=5, backend='xgboost', params=None, rounds=200,
                 surface_col='Track_Turf', distance_col=None,
                 min_races_per_bin=200, use_surface=True):
        self.cv = cv
        self.backend = backend
        self.params = params
        self.rounds = rounds
        self.surface_col = surface_col
        self.distance_col = distance_col
        self.min_races_per_bin = min_races_per_bin
        self.use_surface = use_surface

        self.T_map = None
        self.booster = None
        self.imp_df = None
        self._field_bins = (2, 6, 9, 99)

    def fit(self, X, y):
        print("Starting K-fold training...")
        oof_raw, oof_true = softmax_cv(
            X, y, cv=self.cv, backend=self.backend, params=self.params, rounds=self.rounds
        )
        print("Training final iteration of model...")
        y_win = (y == 1).astype(int)
        self.booster, _, _, self.imp_df = softmax_logistic_model(
            X, X.iloc[:0], y_win, y_win[:0],
            backend=self.backend, params=self.params, rounds=self.rounds, displayFeature=True
        )
        # Learn binned temperatures
        race_meta, sizes = build_race_meta_from_X(
            X, surface_col=self.surface_col, distance_col=self.distance_col, field_bins=self._field_bins
        )
        self.T_map = learn_temperature_multi_binned(
            oof_raw, oof_true, sizes, race_meta,
            min_races_per_bin=self.min_races_per_bin, use_surface=self.use_surface
        )
        return self

    def predict_proba(self, X):
        if self.backend == 'xgboost':
            dte = xgb.DMatrix(X, enable_categorical=True)
            raw = self.booster.predict(dte)
        elif self.backend == 'lightgbm':
            raw = self.booster.predict(X)
        else:
            raise ValueError("backend must be 'xgboost' or 'lightgbm'")

        race_meta, sizes = build_race_meta_from_X(
            X, surface_col=self.surface_col, distance_col=self.distance_col, field_bins=self._field_bins
        )
        p_cal = apply_temperature_binned(
            raw, sizes, race_meta, self.T_map, use_surface=self.use_surface
        )
        return pd.DataFrame(p_cal, index=X.index, columns=["Y_prob"])


class OffsetModel:
    """
    Offset model: market odds as base_margin, XGBoost learns corrections only.

    Uses logit(Implied_Prob) as a fixed offset. XGBoost predicts binary:logistic
    with base_margin, so it only learns adjustments to the market probability.

    Parameters
    ----------
    cv : int
        Not used for CV (trains single model). Kept for API compatibility.
    backend : str
        Only 'xgboost' supported.
    params : dict
        XGBoost parameters.
    rounds : int
        Boosting rounds.
    market_col : str
        Name of the market probability column to use as offset.
    """
    # Default params tuned specifically for offset model (deep_slow config)
    DEFAULT_PARAMS = {
        'learning_rate': 0.01, 'max_depth': 5, 'min_child_weight': 20,
        'subsample': 0.7, 'colsample_bytree': 0.7, 'gamma': 0.2,
        'reg_alpha': 2.0, 'reg_lambda': 10.0,
    }

    def __init__(self, cv=3, backend='xgboost', params=None, rounds=400,
                 market_col='Implied_Prob'):
        self.backend = backend
        # Merge user params on top of offset defaults
        merged = dict(self.DEFAULT_PARAMS)
        if params:
            merged.update(params)
        self.params = merged
        self.rounds = rounds
        self.market_col = market_col
        self.booster = None
        self.imp_df = None
        self._edge_features = None

    def _compute_offset(self, X):
        """Extract logit(market_prob) from X and return edge features + offset."""
        if self.market_col in X.columns:
            impl = X[self.market_col].clip(1e-8, 1 - 1e-8)
            offset = np.log(impl / (1 - impl))
            edge_X = X.drop(columns=[self.market_col, 'log_implied_prob'] if 'log_implied_prob' in X.columns else [self.market_col])
        elif 'log_implied_prob' in X.columns:
            log_impl = X['log_implied_prob']
            impl = np.exp(log_impl)
            impl = impl / impl.groupby(level=0).transform('sum')
            impl = impl.clip(1e-8, 1 - 1e-8)
            offset = np.log(impl / (1 - impl))
            edge_X = X.drop(columns=['log_implied_prob'])
        else:
            raise ValueError("No market probability column found")
        return edge_X, offset

    def fit(self, X, y):
        print("Training offset model (market as base_margin)...")
        edge_X, offset = self._compute_offset(X)
        self._edge_features = list(edge_X.columns)

        y_win = (y == 1).astype(int)

        # Set objective for binary classification with offset
        params = dict(self.params)
        params['objective'] = 'binary:logistic'
        params['eval_metric'] = 'logloss'

        dtrain = xgb.DMatrix(edge_X, label=y_win, base_margin=offset)
        self.booster = xgb.train(
            params, dtrain,
            num_boost_round=self.rounds,
            evals=[(dtrain, 'train')],
            verbose_eval=100,
        )

        # Feature importance
        fi = self.booster.get_score(importance_type='gain')
        if fi:
            self.imp_df = pd.DataFrame(
                list(fi.items()), columns=['feature', 'importance']
            ).sort_values('importance', ascending=False)

        return self

    def predict_proba(self, X):
        edge_X, offset = self._compute_offset(X)

        # Ensure same features as training
        for col in self._edge_features:
            if col not in edge_X.columns:
                edge_X[col] = 0
        edge_X = edge_X[self._edge_features]

        dtest = xgb.DMatrix(edge_X, base_margin=offset)
        preds = self.booster.predict(dtest)

        # Renormalize within race
        preds_series = pd.Series(preds, index=X.index)
        race_sums = preds_series.groupby(level=0).transform('sum')
        preds_norm = preds_series / race_sums

        return pd.DataFrame({'Y_prob': preds_norm}, index=X.index)


# ---------------------------
# Legacy simple temperature (kept for compatibility)
# ---------------------------

def learn_temperature_multi(oof_raw, true_labels, group_sizes):
    """
    Legacy: learn a single global temperature (kept for backward compatibility).
    Prefer learn_temperature_multi_binned + apply_temperature_binned.
    """
    group_ptr = np.cumsum(np.concatenate([[0], group_sizes]))

    def neg_ll(T):
        T = T[0]
        total_ll = 0.0
        num_samples = 0
        for i in range(len(group_ptr) - 1):
            s, e = group_ptr[i], group_ptr[i + 1]
            if e - s <= 1:
                continue
            raw_group = oof_raw[s:e]
            labels_group = true_labels[s:e]
            winner_idxs = np.where(labels_group == 1)[0]
            if len(winner_idxs) == 0:
                continue
            for winner_idx in winner_idxs:
                logits_t = raw_group / T
                p_group = softmax(logits_t)
                total_ll += np.log(p_group[winner_idx] + 1e-12) / len(winner_idxs)
            num_samples += (e - s)
        return -total_ll / num_samples if num_samples > 0 else 0.0

    res = minimize(neg_ll, x0=[1.0], bounds=[(0.05, 10.0)])
    return res.x[0]


def apply_temperature_multi(raw_preds, T, group_sizes):
    """
    Legacy: apply a single global temperature (kept for backward compatibility).
    Prefer apply_temperature_binned.
    """
    group_ptr = np.cumsum(np.concatenate([[0], group_sizes]))
    p_cal = np.empty_like(raw_preds)
    for i in range(len(group_ptr) - 1):
        s, e = group_ptr[i], group_ptr[i + 1]
        raw_group = raw_preds[s:e]
        logits_t = raw_group / T
        p_cal[s:e] = softmax(logits_t)
    return p_cal