import time
import json
from datetime import datetime, timedelta

import os
import pandas as pd
import numpy as np
import re

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def generate_datetime(year):
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    return [(start_date + timedelta(days=i)).strftime("%d/%m/%Y") for i in range((end_date - start_date).days + 1)]

def to_json(dict, filename):
    try:
        with open(filename, 'w') as json_file:
            json.dump(dict, json_file, indent=4)
        print(f"Dictionary successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving dictionary to JSON: {e}")

def save_nested_dict_by_year(horses):
    grouped_horses = {}

    for link, dataframe in horses.items():
        try:
            year = link.split("HK_")[1][:4]  
            if year not in grouped_horses:
                grouped_horses[year] = {}
            grouped_horses[year][link] = dataframe
        except Exception as e:
            print(f"Error processing link: {link}, {e}")
            continue

    for year, data in grouped_horses.items():
        filename = f"horses_HK_{year}.json"
        try:

            json_serializable_data = {
                link: df.to_dict(orient="records") if isinstance(df, pd.DataFrame) else df
                for link, df in data.items()
            }
            with open(filename, "w") as json_file:
                json.dump(json_serializable_data, json_file, indent=4)
            print(f"Saved {len(data)} entries to {filename}")
        except Exception as e:
            print(f"Error saving data for year {year} to {filename}: {e}")

def regularize_dlink(df):
    """
    Normalize all URL columns in a DataFrame so that any variation of the path segment between
    "/racing/information/" and the next slash is replaced with "English", ensuring the base URL
    follows the format:
    https://.../racing/information/English/...

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with normalized URL columns.
    """
    if not isinstance(df, pd.DataFrame):
        print("Provided input is not a DataFrame. Skipping regularization.")
        return df

    df = df.copy()

    url_columns = [col for col in df.columns if 'link' in col.lower()]

    pattern = re.compile(r"(/racing/information/)[^/]+/", flags=re.IGNORECASE)

    for col in url_columns:
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda url: pattern.sub(r"\1English/", url))
        )

    return df

def regularize_rdict(rdict):
    """
    Regularizes the rdict dictionary 
    - Fixing capitalization inconsistencies in horse links (e.g., 'ENGLISH' → 'English').
    - Ensuring date format consistency (removing leading zeros).
    - Expanding two-digit years in 'Date' columns to four-digit years.

    Parameters:
        rdict (dict): A dictionary where keys are horse links, 
                      and values are DataFrames with racing data.

    Returns:
        dict: A new dictionary with regularized keys and date formats.
    """
    normalized_rdict = {}

    for key, df in rdict.items():
        normalized_key = re.sub(r"(/racing/information/)([A-Z]+)/", r"\1English/", key)

        if isinstance(df, pd.DataFrame) and "Date" in df.columns:
            df = df.copy() 
            df["Date"] = df["Date"].str.replace(r"(\d{1,2}/\d{1,2}/)(\d{2})$", r"\g<1>20\g<2>", regex=True)

        normalized_rdict[normalized_key] = df

    return normalized_rdict

def load_pool_dividends(pool_type="WIN", folder_path="../data/historical-data/dividends/"):
    """
    Load dividend data from flat CSVs (post-migration format).

    Parameters
    ----------
    pool_type : str
        The pool type to extract (e.g., 'WIN', 'PLACE'). Default is 'WIN'.
    folder_path : str
        Directory containing dividends-YYYY.csv files.

    Returns
    -------
    dict
        Dictionary with keys as race_id (YYYYMMDDRR) and values as lists of
        dicts with keys: Pool, Winning Combination, Dividend.
    """
    frames = []
    for file in sorted(os.listdir(folder_path)):
        if file.lower().startswith("dividends-") and file.endswith(".csv"):
            full_path = os.path.join(folder_path, file)
            frames.append(pd.read_csv(full_path, dtype={'race_id': str}))

    if not frames:
        print(f"No dividend CSVs found in {folder_path}")
        return {}

    df = pd.concat(frames, ignore_index=True)
    df = df[df['pool'] == pool_type]

    filtered_data = {}
    for _, row in df.iterrows():
        rid = str(row['race_id'])
        entry = {
            'Pool': row['pool'],
            'Winning Combination': str(row['combination']),
            'Dividend': str(row['dividend']),
        }
        filtered_data.setdefault(rid, []).append(entry)

    return filtered_data

def conv_dict(data):
    """
    Converts win_data dict to a DataFrame with race_id, horse_number, and normalized Returns.
    """
    rows = []
    for race_id, entries in data.items():
        for entry in entries:
            if ',' not in entry['Winning Combination']:  
                try:
                    horse_number = int(entry['Winning Combination'])
                    dividend = float(entry['Dividend'].replace(',', '')) / 10
                    rows.append({
                        'race_id': race_id,
                        'Horse No.': horse_number,
                        'Returns': dividend
                    })
                except Exception as e:
                    print(f"Skipping bad entry in race {race_id}: {entry} ({e})")
    return pd.DataFrame(rows).set_index(['race_id', 'Horse No.']).sort_index()

def nested_dict_to_json(data, filename):

    def prepare_data(value):
        """
        Recursively prepare nested dictionary data for JSON serialization.
        Convert DataFrame to dict and Series to a single-column DataFrame dict.
        """

        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        elif isinstance(value, pd.Series):
            return pd.DataFrame(value).reset_index().to_dict(orient="records")
        elif isinstance(value, dict):
            return {k: prepare_data(v) for k, v in value.items()}
        else:
            return value

    try:
        json_serializable_data = prepare_data(data)

        with open(filename, "w") as json_file:
            json.dump(json_serializable_data, json_file, indent=4)
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to JSON: {e}")

def read_nested_dict(filename):

    def reconstruct_data(value):
        """
        Recursively reconstructs data into its original format.
        Converts lists of records back into DataFrames where applicable.
        """
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            try:
                return pd.DataFrame(value) 
            except ValueError:
                return value  
        elif isinstance(value, dict):
            return {k: reconstruct_data(v) for k, v in value.items()}
        else:
            return value

    try:
        with open(filename, "r") as json_file:
            data = json.load(json_file)
        return reconstruct_data(data)
    except Exception as e:
        print(f"Error reading nested dictionary from JSON: {e}")
        return {}

class Utils:
    @staticmethod
    def wait_for_element(driver, by, value, condition=EC.presence_of_element_located, timeout=10):
        return WebDriverWait(driver, timeout).until(condition((by, value)))

    @staticmethod
    def wait_for_page_render(driver, timeout=10):
        print("Waiting page to render...")
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )
            return True
        except Exception as e:
            print(f"Page load timed out: {e}")
            return False
        
    @staticmethod       
    def wait_for_element_with_text(driver, value_of_id, timeout=10):

        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, value_of_id))
        )
        WebDriverWait(driver, timeout).until(
            lambda driver: driver.find_element(By.ID, value_of_id).text.strip() != ""
        )
        return driver.find_element(By.ID, value_of_id)
    
def sort_multiindexed_df(df):
    """
    Sort a DataFrame with a MultiIndex of (race_id, Horse No.) 
    in ascending order for both levels.
    
    Parameters:
    df (pd.DataFrame): DataFrame with MultiIndex (race_id, Horse No.)

    Returns:
    pd.DataFrame: Sorted DataFrame
    """
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError("DataFrame must have a MultiIndex")
    if df.index.names != ['race_id', 'Horse No.']:
        raise ValueError("MultiIndex must be named ['race_id', 'Horse No.']")

    return df.sort_index(level=['race_id', 'Horse No.'], ascending=[True, True])