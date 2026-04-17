"""
model.py
========
Data utilities for splitting and cleaning race data.

Kelly criterion and bankroll simulation live in betting.py.
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))

import warnings
warnings.filterwarnings("ignore")


def remove_tied_entries(df, pos_col='y.status_place'):
    """
    Finds and removes races whose finishing-position labels have gaps or duplicates.

    Parameters:
      df      : pd.DataFrame, MultiIndexed by (race_id, horse_number)
      pos_col : str, name of the integer position column

    Returns:
      df_clean : pd.DataFrame with tied/gappy races removed
    """
    problems = []
    for race_id, grp in df.groupby(level=0):
        found    = np.sort(grp[pos_col].astype(int).unique())
        expected = np.arange(found.min(), found.max() + 1)
        if not np.array_equal(found, expected):
            problems.append(race_id)

    if not problems:
        print("All races have contiguous position labels.")
        return df.copy()

    print(f"Found {len(problems)} problematic races — removing.")
    mask = ~df.index.get_level_values(0).isin(problems)
    return df.loc[mask].copy()


def split_data(df, train_years, test_year, y_col, remove_ent=False, shuffle=False):
    """
    Split into train/test by year. Drops void entries from training set.
    """
    from sklearn.model_selection import train_test_split

    train_df = df[df['Year'].isin(train_years)].copy()
    test_df = df[df['Year'].isin(test_year)].copy()

    train_df = train_df[train_df['y.status_place'] != 99].copy()
    if remove_ent:
        train_df = remove_tied_entries(train_df)

    X_train = train_df.drop(columns=[y_col, 'Year'])
    Y_train = train_df[y_col]
    X_test = test_df.drop(columns=[y_col, 'Year'])
    Y_test = test_df[y_col]

    if shuffle:
        X_train, _, Y_train, _ = train_test_split(
            train_df.drop(columns=[y_col, 'Year']), train_df[y_col],
            test_size=0.2, random_state=42, shuffle=True,
        )
        X_test, _, Y_test, _ = train_test_split(
            test_df.drop(columns=[y_col, 'Year']), test_df[y_col],
            test_size=0.2, random_state=42, shuffle=True,
        )

    return X_train, X_test, Y_train, Y_test
