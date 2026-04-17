import re   
from datetime import datetime

from utilities import *
import time

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC


class LiveScraper:
    ODDS_TYPE_MAP = {
        "cross_alup": "All Up/ Cross Pool All Up",
        "wp": "Win / Place",
        "wpq": "Quinella / Quinella Place",
        "fct": "Forecast",
        "tce": "Trio",
        "tri": "Trifecta",
        "ff": "First Four",
        "qtt": "Quartet",
        "dbl": "Double",
        "tbl": "Treble",
        "dt": "Daily Triple",
        "tt": "Triple Trio",
        "6up": "Six Up",
        "jkc": "Jockey Challenge",
        "tnc": "Trainer Challenge",
        "jtcombo": "Jockey Trainer Combo",
        "pwin": "Progressive Win Odds",
        "turnover": "Pool Investment"
    }


    def __init__(self, driver):
        self.driver = driver
        self.current_page = None
        self.odd_type = None
        self.total_pages = None
        self.base_url = "https://bet.hkjc.com/en/racing/"

    def _initialize_navigation_state(self):
        """
        Initialize the navigation state by detecting:
        - Total number of pages (races).
        - The current page (race number).
        - The current odds type (e.g., 'wp', 'wpq').

        Returns:
            bool: True if initialization was successful, False otherwise.
        """
        try:
            race_buttons = self.driver.find_elements(By.XPATH, "//div[contains(@id, 'raceno_')]")
            self.total_pages = len(race_buttons)
            print(f"Total pages detected: {self.total_pages}")

            try:
                meeting_desc_no = Utils.wait_for_element(self.driver, By.ID, "meetingDescNo").text
                if "Race" in meeting_desc_no:
                    self.current_page = int(meeting_desc_no.split("Race")[-1].strip())
                    print(f"Current page detected: {self.current_page}")
                else:
                    self.current_page = None
                    print("No race currently detected.")
            except Exception as e:
                print(f"Error detecting current page: {e}")
                self.current_page = None

            try:
                list_items = self.driver.find_elements(By.TAG_NAME, "li")  
                self.odd_type = None  

                for list_item in list_items:
                    odd_id = list_item.get_attribute("id")
                    element_class = list_item.get_attribute("class")
                    if odd_id in self.ODDS_TYPE_MAP and "active" in element_class:
                        self.odd_type = odd_id
                        print(f"Current active odds type detected:({self.ODDS_TYPE_MAP.get(self.odd_type)})")
                        break

                if not self.odd_type:
                    print("No active odds type detected. Reverting to None.")
            except Exception as e:
                print(f"Error detecting odds type: {e}")
                self.odd_type = None
            return True

        except Exception as e:
            print(f"Error occurred during navigation state initialization: {e}")
            return False

    def navigate_to_page(self, race_number):
        """
        Navigate to a specific race number on the HKJC page.

        Parameters:
            race_number (int): The race number to navigate to (e.g., 2 for Race 2).

        Returns:
            bool: True if navigation is successful, False otherwise.
        """
        try:
            if self.current_page == race_number:
                print(f"Already on Race {race_number}. No navigation required.")
                return True

            race_id = f"raceno_{race_number}"
            race_button = Utils.wait_for_element(self.driver, By.ID, race_id)

            ActionChains(self.driver).move_to_element(race_button).perform()
            race_button.click()

            if not Utils.wait_for_page_render(self.driver):
                print(f"Page did not finish rendering after navigating to Race {race_number}.")
                return False

            self.current_page = race_number
            print(f"Successfully navigated to Race {race_number}.")
            return True

        except Exception as e:
            print(f"Error occurred while navigating to Race {race_number}: {e}")
            return False

    def navigate_to_odds_type(self, odd_type):

        try:
            if self.odd_type == odd_type:
                print(f"Already on odds type '{self.ODDS_TYPE_MAP[odd_type]}'. No navigation required.")
                return True

            odd_type_button = Utils.wait_for_element(self.driver, By.ID, odd_type)
            odd_type_button.click()

            if not Utils.wait_for_page_render(self.driver):
                print(f"Page did not finish rendering after navigating to '{self.ODDS_TYPE_MAP[odd_type]}'.")
                return False

            self.odd_type = odd_type
            print(f"Successfully navigated to '{self.ODDS_TYPE_MAP[odd_type]}'. Initializing state...")

            self._initialize_navigation_state()

            return True

        except Exception as e:
            print(f"Error occurred while navigating to '{self.ODDS_TYPE_MAP[odd_type]}': {e}")
            return False


    def refresh_page(self):
        try:
            refresh_button = Utils.wait_for_element(self.driver, By.ID, "refreshButton")
            refresh_button.click()

            if not Utils.wait_for_page_render(self.driver):
                print("Page did not finish rendering after clicking refresh.")
                return None
            refresh_time_element = Utils.wait_for_element(self.driver, By.ID, "refreshTime")
            last_update_time = refresh_time_element.text.replace("Last Update: ", "").strip()

            try:
                refresh_datetime = datetime.strptime(last_update_time, "%d/%m/%Y %H:%M")
                print(f"Parsed last update time as datetime: {refresh_datetime}")
                return refresh_datetime
            except ValueError:
                print(f"Failed to parse last update time: {last_update_time}")
                return None

        except Exception as e:
            print(f"Error occurred while refreshing and getting update time: {e}")
            return None


    def scrape_meeting_info(self):

        try:
            banner = Utils.wait_for_element(self.driver, By.ID,
                                            "meetingDescDtls").text
            print("Scraped race information successfully.")
            print(f"Meeting Info: {banner}")

            parts = [p.strip() for p in banner.split(",")]

            if len(parts) < 3:
                raise ValueError("Unexpected banner format")

            date_part = parts[0]         
            time_part = parts[2]          

            if re.match(r"\d{2}/\d{2}/\d{4}$", date_part):
                full_date = date_part                              
            else:
                m = re.search(r"/(\d{4})-(\d{2})-(\d{2})/", self.driver.current_url)
                if m:
                    year = m.group(1)
                    full_date = f"{date_part}/{year}"             
                else:
                    full_date = f"{date_part}/{datetime.now().year}"

            match_dt = datetime.strptime(f"{full_date} {time_part}",
                                        "%d/%m/%Y %H:%M")

            return {"meeting_info_content": banner,
                    "match_datetime":        match_dt}

        except Exception as e:
            print(f"Failed to parse datetime from meeting info: {e}")
            return {"meeting_info_content": banner,
                    "match_datetime":        None}
        
    def scrape_odds_table_type_wp(self):

        try:
            odds_table = self.driver.find_element(By.XPATH, "//div[@id='rcOddsTable']//table")
            header_row = odds_table.find_element(By.XPATH, ".//tr[1]")            
            headers = [header.text.strip() for header in header_row.find_elements(By.XPATH, "//td[@rowspan='1']")]
            
            rows = odds_table.find_elements(By.XPATH, "./tr[./*[@id]]")

            print(f"Number of rows to scrape: {len(rows)}")

            data = []
            for row_number, row in enumerate(rows, start=1):
                try:
                    row_data = {}

                    for header in headers: 
                        if header == "No.":
                            row_data[header] = odds_table.find_element(By.ID, f"runnerNo_{self.current_page}_{row_number}").text
                        elif header == "Horse Name":
                            row_data[header] = odds_table.find_element(By.ID, f"horseName_{self.current_page}_{row_number}").text
                        elif header == "Draw":
                            row_data[header] = odds_table.find_element(By.ID, f"draw_{self.current_page}_{row_number}").text
                        elif header == "Wt.":
                            row_data[header] = odds_table.find_element(By.ID, f"handicapWt_{self.current_page}_{row_number}").text
                        elif header == "Jockey":
                            row_data[header] = odds_table.find_element(By.ID, f"jockey_{self.current_page}_{row_number}").text
                        elif header == "Trainer":
                            row_data[header] = odds_table.find_element(By.ID, f"trainer_{self.current_page}_{row_number}").text
                        elif header == "Win":
                            row_data[header] = odds_table.find_element(By.ID, f"odds_WIN_{self.current_page}_{row_number}").text
                        elif header == "Place":
                            row_data[header] = odds_table.find_element(By.ID, f"odds_PLA_{self.current_page}_{row_number}").text

                    data.append(row_data)

                except Exception as e:
                    print(f"Error scraping row {row_number}: {e}")

            df = pd.DataFrame(data)
            print("Scraped DataFrame: \n", df)
            return df

        except Exception as e:
            print(f"Error occurred while scraping the Win / Place odds table: {e}")
            return pd.DataFrame()
        
    def scrape_investment_table_type_wp(self):
        try:
            pool_ids = {
                "Win Pool": "poolInvWIN",
                "Place Pool": "poolInvPLA",
                "Quinella Pool": "poolInvQIN",
                "Quinella Place Pool": "poolInvQPL",
            }

            investment_data = {}
            for pool_name, pool_id in pool_ids.items():
                try:
                    element = self.driver.find_element(By.ID, pool_id)                    
                    tds = element.find_elements(By.TAG_NAME, "td")
                    if len(tds) > 1:
                        raw_text = tds[1].text.strip()
                        numeric_value = raw_text.replace("$", "").replace(",", "").strip()
                        investment_data[pool_name] = numeric_value if numeric_value.isdigit() else None
                    else:
                        investment_data[pool_name] = None
                except Exception as e:
                    print(f"Could not extract data for {pool_name}: {e}")
                    investment_data[pool_name] = None

            print("Investment Data:")
            for key, value in investment_data.items():
                print(f"{key}: {value}")
            return investment_data

        except Exception as e:
            print(f"Error occurred while scraping the investment table... ID's might not exist...: {e}")
            return {pool_name: None for pool_name in pool_ids.keys()}
        
    def scrape_page_type_wp(self):
        try:
            print(f"Scraping odds table for race {self.current_page}...")
            odds_table = self.scrape_odds_table_type_wp()
            if odds_table.empty:
                print(f"Failed to scrape odds table for race {self.current_page}. Aborting scrape.")
                return pd.DataFrame()

            print(f"Scraping investment table for race {self.current_page}...")
            investment_data = self.scrape_investment_table_type_wp()
            if not investment_data:
                print(f"Failed to scrape investment table for race {self.current_page}. Aborting scrape.")
                return pd.DataFrame()

            for key, value in investment_data.items():
                odds_table[key] = value

            print(f"Scrape completed successfully for race {self.current_page}. Consolidated DataFrame:")
            return odds_table

        except Exception as e:
            print(f"Error occurred while scraping race {self.current_page}: {e}")
            return pd.DataFrame()

    def scrape_page(self):
        try:
            refresh_datetime = self.refresh_page()
            if not refresh_datetime:
                print("Failed to refresh the page and get update time. Aborting scrape.")
                return pd.DataFrame()

            meeting_info = self.scrape_meeting_info()
            if not meeting_info:
                print("Failed to scrape meeting info. Aborting scrape.")
                return pd.DataFrame()

            match_datetime = meeting_info["match_datetime"]

            if self.odd_type == 'wp':
                print("Scraping Win/Place page...")
                df = self.scrape_page_type_wp()
            elif self.odd_type == 'wpq':
                print("Scraping Quinella/Quinella Place page...")
                df = self.scrape_page_type_wpq()
            else:
                print(f"Unsupported page type: {self.odd_type}. Aborting scrape.")
                return pd.DataFrame()

            if df.empty:
                print(f"Failed to scrape data for page type '{self.odd_type}'. Aborting scrape.")
                return pd.DataFrame()

            df["race_number"] = self.current_page
            df["match_datetime"] = match_datetime
            df["odds_datetime"] = refresh_datetime
            df["meeting_info"] = meeting_info["meeting_info_content"]

            metadata_columns = ["race_number", "match_datetime", "odds_datetime", "meeting_info"]
            df = df[metadata_columns + [col for col in df.columns if col not in metadata_columns]]
            print("Scrape completed successfully. Consolidated DataFrame: \n", df)
            return df

        except Exception as e:
            print(f"Error occurred while scraping the page: {e}")
            return pd.DataFrame()

    def scrape_page_type_wpq(self):
        """
        Scrape the Quinella / Quinella Place (wpq) grid for the *current* race.

        Returns
        -------
        pd.DataFrame
            Columns:
                Horse1, Horse2, Quinella, Quinella Place,
                Win Pool, Place Pool, Quinella Pool, Quinella Place Pool
            One row per valid horse pair (x < y).
        """
        try:
            Utils.wait_for_element(self.driver, By.ID, "qb_QIN_1_2")

            qin_cells = self.driver.find_elements(
                By.XPATH, "//td[starts-with(@id,'qb_QIN_')]")
            qpl_cells = self.driver.find_elements(
                By.XPATH, "//td[starts-with(@id,'qb_QPL_')]")

            def _parse_pair(cell_id: str):
                _, _, h1, h2 = cell_id.split("_")
                return int(h1), int(h2)

            odds_map = {}          

            for cell in qin_cells:
                h1, h2 = _parse_pair(cell.get_attribute("id"))
                if h1 >= h2:          
                    continue
                odds_map.setdefault((h1, h2), {})["Quinella"] = cell.text.strip() or None

            for cell in qpl_cells:
                h1, h2 = _parse_pair(cell.get_attribute("id"))
                if h1 >= h2:
                    continue
                odds_map.setdefault((h1, h2), {})["Quinella Place"] = cell.text.strip() or None

            if not odds_map:
                print(f"No Quinella / Quinella Place odds found for race {self.current_page}.")
                return pd.DataFrame()

            records = []
            for (h1, h2), odds in sorted(odds_map.items()):
                records.append({
                    "Horse_Number_1": h1,
                    "Horse_Number_2": h2,
                    "Quinella": odds.get("Quinella"),
                    "Quinella Place": odds.get("Quinella Place"),
                })

            df = pd.DataFrame.from_records(records)

            print(f"Scraped Quinella/Quinella Place odds for race {self.current_page}")
            print(df.head())
            return df

        except Exception as e:
            print(f"Error occurred while scraping Quinella / Quinella Place odds: {e}")
            return pd.DataFrame()

    def scrape_all_games(self, odds_type):
        try:
            if not self.navigate_to_odds_type(odds_type):
                print("Failed to navigate to 'Win/Place'. Aborting scrape.")
                return pd.DataFrame()
            
            race_numbers = range(1, self.total_pages + 1)
            all_race_data = []

            for race_number in race_numbers:
                print(f"Scraping Race {race_number}...")

                if not self.navigate_to_page(race_number):
                    print(f"Failed to navigate to Race {race_number}. Skipping.")
                    continue
                
                Utils.wait_for_page_render(self.driver)
                print("Sleeping one second...")
                time.sleep(1)
                
                Utils.wait_for_element_with_text(self.driver, odds_type)

                race_data = self.scrape_page()
                if race_data.empty:
                    print(f"Failed to scrape data for Race {race_number}. Skipping.")
                    continue

                all_race_data.append(race_data)

            if all_race_data:
                final_df = pd.concat(all_race_data, ignore_index=True)
                print("Scraping completed successfully. Consolidated DataFrame:")
                print(final_df)
                return final_df
            else:
                print("No race data was successfully scraped. Returning an empty DataFrame.")
                return pd.DataFrame()

        except Exception as e:
            print(f"Error occurred while scraping all games: {e}")
            return pd.DataFrame()

    def scrape_single_race(self, odds_type: str, race_number: int) -> pd.DataFrame:
        """
        Scrape ONE race for ONE odds type.

        Parameters
        ----------
        odds_type : str
            Tab code – e.g. "wp" (Win/Place) or "wpq" (Quinella / Quinella Place)
        race_number : int
            1-based race number as shown on the HKJC site.

        Returns
        -------
        pd.DataFrame
            • For 'wp' → includes investment columns  
            • For 'wpq' → odds only (no investment columns)  
            Returns an empty DataFrame on any failure.
        """
        if odds_type not in self.ODDS_TYPE_MAP:
            raise ValueError(f"Unknown odds type: {odds_type!r}")

        if not self.navigate_to_odds_type(odds_type):
            print(f"Could not navigate to odds type '{odds_type}'.")
            return pd.DataFrame()

        if not self.navigate_to_page(race_number):
            print(f"Could not navigate to Race {race_number}.")
            return pd.DataFrame()

        Utils.wait_for_page_render(self.driver)
        time.sleep(1)
        Utils.wait_for_element_with_text(self.driver, odds_type)

        return self.scrape_page()

# if __name__ == '__main__':  
#     driver = webdriver.Chrome()
#     scraper = LiveScraper(driver)

#     driver.get(scraper.base_url)
#     scraper.scrape_all_games('wp')