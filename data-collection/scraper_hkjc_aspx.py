import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../data-collection")))
from utilities import *

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException

from webdriver_manager.chrome import ChromeDriverManager
import time

class ASPXScraper:
    def __init__(self, driver):
        self.driver = driver
        self.total_pages = None
        self.current_page = None
        self.simulcast_detected = False

    def simulcast_page(self):
        try:
            simulcast_element = self.driver.find_element(By.ID, "simSubNavButton")
            if simulcast_element:
                print("Simulcast page detected.")
                return True
        except Exception:
            pass 
        return False

    def _initialize_navigation_state(self):
        """
        Detect total pages, current page, and simulcast state.
        This does not handle dates or 'All Races'.
        """

        try:
            race_table = Utils.wait_for_element(self.driver, By.CLASS_NAME, "f_fs12.js_racecard").find_element(By.TAG_NAME, "tbody")
            tr_tags = race_table.find_elements(By.TAG_NAME, "tr")

            if len(tr_tags) == 2:
                print("Simulcast detected...")
                self.simulcast_detected = True
            elif len(tr_tags) != 1:
                print("Unexpected table structure.")
                return False

            td_tags = tr_tags[0].find_elements(By.TAG_NAME, "td")
            total_races = 0
            self.current_page = None

            for index, td in enumerate(td_tags):
                link = td.find_element(By.TAG_NAME, "a") if td.find_elements(By.TAG_NAME, "a") else None
                img = td.find_element(By.TAG_NAME, "img") if td.find_elements(By.TAG_NAME, "img") else None

                if link:
                    href = link.get_attribute("href")
                    if "RaceNo=" in href:
                        race_no = href.split("RaceNo=")[-1]
                        if race_no.isdigit():
                            total_races += 1
                elif img:
                    img_src = img.get_attribute("src")
                    if f"racecard_rt_" in img_src and "_o.gif" in img_src:
                        try:
                            num = int(img_src.split("racecard_rt_")[1].split("_o.gif")[0])
                            self.current_page = num
                            total_races += 1
                            print(f"Current page detected as {self.current_page}.")
                        except ValueError:
                            print(f"Failed to extract page number from {img_src}")

            self.total_pages = total_races
            print(f"Total races detected: {total_races}")
            return True
        except Exception as e:
            print(f"Error during navigation state initialization: {e}")
            return False
        
    def return_all_dates(self):
        try:
            select_element = Utils.wait_for_element(self.driver, By.ID, "selectId")
            date_select = Select(select_element)
            available_dates = [option.get_attribute('value') for option in date_select.options]
            print(f"Available dates: {available_dates}")
            return available_dates
        except Exception as e:
            print(f"Error occurred while fetching available dates: {e}")
            return []
        
    def navigate_to_page(self, page_number):
        """
        Navigate to a specific race page or "All Races" on the HKJC site.

        Parameters:
            page_number (int): The page number to navigate to (1-based index). Use 0 for "All Races".

        Returns:
            bool: True if navigation is successful, False otherwise.
        """
        try:
            if page_number < 0 or page_number > self.total_pages:
                print(f"Invalid page_number: {page_number}. Available pages: 0 to {self.total_pages}.")
                return False

            if page_number == self.current_page:
                print(f"Already on page {page_number}. No navigation required.")
                return True

            race_table = Utils.wait_for_element(self.driver, By.CLASS_NAME, "f_fs12.js_racecard").find_element(By.TAG_NAME, "tbody")
            td_tags = race_table.find_elements(By.TAG_NAME, "td")
            target_td = None

            if page_number == 0:  
                for td in td_tags:
                    img = td.find_element(By.TAG_NAME, "img") if td.find_elements(By.TAG_NAME, "img") else None
                    if img and img.get_attribute("src").endswith("racecard_rt_all.gif"):
                        target_td = td
                        break
                if not target_td:
                    print("Failed to locate 'All Races' button.")
                    return False
                print("Navigating to 'All Races'...")
            else:  
                for td in td_tags:
                    link = td.find_element(By.TAG_NAME, "a") if td.find_elements(By.TAG_NAME, "a") else None
                    if link and f"RaceNo={page_number}" in link.get_attribute("href"):
                        target_td = td
                        break
                if not target_td:
                    print(f"Failed to locate button for page {page_number}.")
                    return False
                print(f"Navigating to page {page_number}...")

            ActionChains(self.driver).move_to_element(target_td).perform()
            target_td.click()

            if not Utils.wait_for_page_render(self.driver):
                print(f"Page did not finish loading after navigating to page {page_number}.")
                return False

            self.current_page = page_number
            print(f"Successfully navigated to {'All Races' if page_number == 0 else f'page {page_number}'}.")
            return True
        except Exception as e:
            print(f"Error occurred during page navigation: {e}")
            return False

class ResultsScraper(ASPXScraper):
    def __init__(self, driver):
        super().__init__(driver)
        self.current_date = None
        self.base_url = "https://racing.hkjc.com/racing/information/English/racing/LocalResults.aspx"

    def _initialize_navigation_state(self):
        """
        Initialize the navigation state, including date, total pages, and current page.
        """
        try:
            if self.simulcast_page():
                print("Simulcast detected during navigation state initialization. Skipping page.")
                return False 

            if not super()._initialize_navigation_state():
                print("Parent navigation state initialization failed.")
                return False
            return True

        except Exception as e:
            print(f"Error during results navigation state initialization: {e}")
            return False

    def validate_date(self):
        """
        Validate if the date displayed on the page matches `self.current_date`.

        Parameters:
            empty_pages (list): A list to store dates where validation fails.

        Returns:
            bool: True if the date is valid, False otherwise.
        """
        try:
            meeting_div = self.driver.find_element(By.CLASS_NAME, "raceMeeting_select")
            meeting_text = meeting_div.text.strip().split("\n")[0]

            match = re.search(r"(\d{2}/\d{2}/\d{4})", meeting_text)
            if not match:
                print(f"Warning: Could not extract date from race meeting text: {meeting_text}")
                return False

            page_date = match.group(1)
            if page_date != self.current_date:
                print(f"Date mismatch detected! Expected {self.current_date}, but found {page_date}. Skipping.")
                return False

            return True

        except Exception as e:
            print(f"Error occurred while validating race meeting date: {e}")
            return False
        
    def validate_page(self):
        """
        Validate the race page by extracting the race number from the page and comparing it to self.current_page.

        Returns:
            bool: True if the extracted race number matches self.current_page, False otherwise.
        """
        try:
            race_tab_div = self.driver.find_element(By.CLASS_NAME, "race_tab")
            race_table = race_tab_div.find_element(By.TAG_NAME, "table")

            race_number_td = race_table.find_element(By.XPATH, ".//td[@colspan]")
            
            match = re.search(r"RACE\s+(\d+)", race_number_td.text)

            if match:
                extracted_race_number = int(match.group(1))
                if extracted_race_number == self.current_page:
                    return True
                else:
                    print(f"Race number mismatch: Expected {self.current_page}, but found {extracted_race_number}.")
                    return False
            else:
                print("No valid race number found in race tab. Skipping...")
                return False

        except Exception as e:
            print(f"Error occurred while validating race page number: {e}")
            return False

    def navigate_to_date(self, date_value: str) -> bool:

        try:

            url_date = datetime.strptime(date_value, "%d/%m/%Y").strftime("%Y/%m/%d")
            target   = f"{self.base_url}?RaceDate={url_date}"
            print(f"👉  {target}")
            self.driver.get(target)

            if not Utils.wait_for_page_render(self.driver):
                print("⚠️  page did not finish rendering.")
                return False


            for bad_id, msg in (("errorContainer", "no meeting"),
                                ("simSubNavButton", "simulcast detected")):
                try:
                    elt = self.driver.find_element(By.ID, bad_id)
                    if elt.is_displayed():
                        print(f"↪  {msg} on {date_value} – skipping.")
                        return False
                except Exception:
                    pass   # element not present → fine


            self.current_date = date_value
            print(f"✅  ready on {date_value}")
            return self._initialize_navigation_state()

        except Exception as e:
            print(f"❌  navigate_to_date({date_value}) failed: {e}")
            return False
            
    def scrape_race_meeting(self):
        try:
            track_div = self.driver.find_element(By.CLASS_NAME, "f_clear.top_races")
            
            track_text = track_div.text.strip() 
            if "Happy Valley" in track_text:
                return "HV"
            elif "Sha Tin" in track_text:
                return "ST"
            else:
                print(f"Unknown track found in text: {track_text}")
                return None
            
        except Exception as e:
            print(f"Error occurred while scraping race track: {e}")
            return None

    def scrape_table_type_performance(self):
        """
        Scrape a table within the 'performance' div and extract data into a DataFrame.

        Returns:
            pandas.DataFrame: DataFrame containing the table data.
        """
        try:
            performance_div = Utils.wait_for_element(self.driver, By.CLASS_NAME, "performance")
            table = performance_div.find_element(By.TAG_NAME, "table")

            header_row = table.find_element(By.TAG_NAME, "thead").find_element(By.TAG_NAME, "tr")
            headers = [td.text.strip() for td in header_row.find_elements(By.TAG_NAME, "td")]

            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            data = []
            for row in rows:
                row_data = {}
                tds = row.find_elements(By.TAG_NAME, "td")
                for header, td in zip(headers, tds):
                    link_element = td.find_element(By.TAG_NAME, "a") if td.find_elements(By.TAG_NAME, "a") else None
                    row_data[header] = td.text.strip()
                    if link_element:
                        row_data[f"{header}_link"] = link_element.get_attribute("href")

                data.append(row_data)

            df = pd.DataFrame(data)
            print("Scraped DataFrame:")
            print(df)
            return df

        except Exception as e:
            print(f"Error occurred while scraping the table: {e}")
            return pd.DataFrame()
        

    def scrape_div_type_race_tab(self):
        try:
            race_tab_div = self.driver.find_element(By.CLASS_NAME, "race_tab")
            tbody       = race_tab_div.find_element(By.TAG_NAME, "tbody")
            rows        = tbody.find_elements(By.TAG_NAME, "tr")
            
            race_details = {
                "Going": None,
                "Course": None,
            }

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) < 2:
                    continue

                label = cells[1].text.lower().strip().rstrip(':').strip()  

                if label == "going":
                    race_details["Going"] = cells[2].text.strip() 
                if label == "course":
                    race_details["Course"] = cells[2].text.strip() 

            print("Extracted race details:")
            print(race_details)
            return race_details

        except Exception as e:
            print(f"Error occurred while scraping race_tab: {e}")
            return {}

        
    def scrape_table_type_dividend(self):

        try:
            dividend_tab = self.driver.find_element(By.CLASS_NAME, "dividend_tab.f_clear")
            table = dividend_tab.find_element(By.TAG_NAME, "table")
            tbody = table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")
            
            data = []
            current_pool = None

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells:
                    continue  

                pool_cell = cells[0]
                if pool_cell.get_attribute("rowspan"):
                    current_pool = pool_cell.text.strip()  
                    winning_combination = cells[1].text.strip()
                    dividend = cells[2].text.strip()
                else:
                    winning_combination = cells[0].text.strip()
                    dividend = cells[1].text.strip()

                data.append({
                    "Pool": current_pool,
                    "Winning Combination": winning_combination,
                    "Dividend": dividend
                })

            df = pd.DataFrame(data)
            print("Scraped Dividends Table:")
            print(df)
            return df

        except Exception as e:
            print(f"Error while scraping the dividend table: {e}")
            return pd.DataFrame()
        
    def scrape_results_page(self):

        try:
            track = self.scrape_race_meeting()
            if not track:
                print("Failed to scrape track location.")

            race_details = self.scrape_div_type_race_tab()
            race_details["Track"] = track

            performance_data = self.scrape_table_type_performance()
            if performance_data.empty:
                print("No performance data found on this page.")
                return {
                    "race_performance_data": pd.DataFrame(),
                    "dividends_data": pd.DataFrame()
                }

            for key, value in race_details.items():
                performance_data[key] = value

            dividends_data = self.scrape_table_type_dividend()

            print("Successfully scraped race page.")

            performance_data = regularize_dlink(performance_data)

            return {
                "race_performance_data": performance_data,
                "dividends_data": dividends_data
            }

        except Exception as e:
            print(f"Error occurred while scraping race page: {e}")
            return {
                "race_performance_data": pd.DataFrame(),
                "dividends_data": pd.DataFrame()
            }
        
    def scrape_results_all_pages(self, date):

        try:
            if not self.navigate_to_date(date):
                print(f"Failed to navigate to date {date}.")
                return {
                    "race_performance_data": pd.DataFrame(),
                    "dividends_data": {},
                    "invalid_pages": []
                }

            print("Sleeping 1 second...")
            time.sleep(1)
            all_race_data = []
            all_dividends = {}
            invalid_pages = []

            for race_number in range(1, self.total_pages + 1):
                try:
                    print(f"Scraping race number {race_number} for date {date}... Sleeping 1 second...")
                    time.sleep(1)

                    if not self.navigate_to_page(race_number):
                        print(f"Failed to navigate to race number {race_number}. Logging as an empty page.")
                        invalid_pages.append(race_number)
                        continue
                    
                    if not self.validate_date():
                        print(f"Page date validation failed for {date}. Logging race {race_number} as invalid.")
                        invalid_pages.append(race_number)
                        continue

                    if not self.validate_page():
                        print(f"Page race number validation failed for race {race_number}. Logging as invalid.")
                        invalid_pages.append(race_number)
                        continue

                    result = self.scrape_results_page()

                    if result["race_performance_data"].empty:
                        print(f"No data found on race number {race_number}. Logging as an empty page.")
                        invalid_pages.append(race_number)
                        continue

                    race_data = result["race_performance_data"]
                    race_data.insert(0, "Date", date) 
                    race_data.insert(1, "race_number", race_number) 
                    all_race_data.append(race_data)

                    if not result["dividends_data"].empty:
                        all_dividends[race_number] = result["dividends_data"]

                except Exception as e:
                    print(f"Error occurred while scraping race number {race_number}: {e}")
                    invalid_pages.append(race_number)

            race_performance_data = pd.concat(all_race_data, ignore_index=True) if all_race_data else pd.DataFrame()

            return {
                "race_performance_data": race_performance_data,
                "dividends_data": all_dividends,
                "invalid_pages": invalid_pages
            }

        except Exception as e:
            print(f"Error occurred while scraping all pages for date {date}: {e}")
            return {
                "race_performance_data": pd.DataFrame(),
                "dividends_data": {},
                "invalid_pages": ["General error"]
            }
        

    def scrape_results_all_pages_datelist(self, datelist):
        """
        Scrape all race pages for a list of dates.

        Parameters:
            datelist (list): List of dates in "dd/mm/yyyy" format.

        Returns:
            dict: Combined race-performance data, nested dividends data by date and race number, and empty pages by date.
                {
                    "race_performance_data": pandas.DataFrame,
                    "dividends_data": dict,
                    "invalid_pages": dict
                }
        """
        all_race_data = []
        all_dividends = {}
        all_invalid_pages = {}

        for date in datelist:
            print(f"\nScraping all pages for date: {date}...")
            try:
                print("Sleeping 1 second....")
                time.sleep(1)
                result = self.scrape_results_all_pages(date)

                if not result["race_performance_data"].empty:
                    all_race_data.append(result["race_performance_data"])

                if result["dividends_data"]:
                    all_dividends[date] = result["dividends_data"]

                if result["invalid_pages"]:
                    all_invalid_pages[date] = result["invalid_pages"]

            except Exception as e:
                print(f"Error occurred while scraping all pages for date {date}: {e}")
                all_invalid_pages[date] = ["General error"]

        combined_race_performance_data = (
            pd.concat(all_race_data, ignore_index=True) if all_race_data else pd.DataFrame()
        )

        return {
            "race_performance_data": combined_race_performance_data,
            "dividends_data": all_dividends,
            "invalid_pages": all_invalid_pages,
        }
    
class HorseScraper(ASPXScraper):
    def scrape_table_type_season_record(self):
        try:
            season_table = self.driver.find_element(By.CLASS_NAME, "bigborder")
            tbody = season_table.find_element(By.TAG_NAME, "tbody")
            rows = tbody.find_elements(By.TAG_NAME, "tr")

            header_cells = rows[0].find_elements(By.XPATH, "./td[not(.//td)]")  
            headers = []
            exclude_column_index = None

            for i, td in enumerate(header_cells):
                header_text = td.text.strip().replace("\n", " ")
                if header_text == "Video Replay":
                    exclude_column_index = i  
                elif header_text:
                    headers.append(header_text)

            data = []

            for row in rows[1:]:
                if row.find_elements(By.XPATH, "./td[@colspan]"):  
                    continue

                row_data = {}
                cells = row.find_elements(By.XPATH, "./td[not(.//td)]") 

                for col_index, (header, cell) in enumerate(zip(headers, cells)):
                    if col_index == exclude_column_index:
                        continue  

                    cell_text = cell.text.strip() if cell.text.strip() != "--" else None
                    row_data[header] = cell_text

                    link_element = cell.find_element(By.TAG_NAME, "a") if cell.find_elements(By.TAG_NAME, "a") else None
                    if link_element:
                        href = link_element.get_attribute("href")
                        if not href.startswith("javascript:"):  
                            row_data[f"{header}_link"] = href

                if any(value for value in row_data.values()):
                    data.append(row_data)

            df = pd.DataFrame(data)

            print("Scraped Season Record Table (without Video Replay column):")
            print(df)
            return df

        except Exception as e:
            print(f"Error occurred while scraping the season record table: {e}")
            return pd.DataFrame()
        
    def scrape_season_records(self, links):

        season_records = {}
        invalid_links = []

        for link in links:
            mod_link = f"{link}&Option=1"

            print(f"Scraping link: {link}")

            try:
                self.driver.get(mod_link)
                time.sleep(1) 
                
                df = self.scrape_table_type_season_record()
                if df.empty:
                    print(f"No data found for link: {link}")
                    invalid_links.append(link)
                else: 
                    season_records[link] = df
            except Exception as e:
                print(f"Error occurred while scraping link {link}: {e}")
                invalid_links.append(link)

        return {
            "season_records": season_records,
            "invalid_links": invalid_links
        }

class RacecardScraper(ASPXScraper):
    def __init__(self, driver):
        super().__init__(driver)
        self.base_url = "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx"

    def scrape_next_race_card(self) -> pd.DataFrame:
        """
        Scrape the next race‑day card, or return empty DataFrame if not yet available.
        """
        try:
            error_div = self.driver.find_element(By.ID, "errorContainer")
            if error_div.is_displayed():
                print("No upcoming race card available → returning empty DataFrame.")
                return pd.DataFrame()
        except NoSuchElementException:
            pass  

        if not self._initialize_navigation_state():
            print("Failed to initialize navigation state. Aborting scrape.")
            return pd.DataFrame()

        aggregated = []
        for page in range(1, self.total_pages + 1):
            if not self.navigate_to_page(page):
                continue
            dt = self._extract_datetime()
            if dt is None:
                continue
            df = self._scrape_table()
            if df.empty:
                continue
            df.insert(0, "race_number", page)
            df.insert(1, "datetime", dt)
            aggregated.append(df)
            time.sleep(1)

        return pd.concat(aggregated, ignore_index=True) if aggregated else pd.DataFrame()

        
    def _extract_datetime(self):
        """
        Extract the datetime from the race card page.

        Returns:
            datetime.datetime: Extracted datetime object, or None if parsing failed.
        """
        try:
            element = Utils.wait_for_element(self.driver, By.CLASS_NAME, "f_fs13")
            full_text = element.get_attribute("innerHTML").split("<br>")[1].strip()
            
            parts = full_text.split(", ")
            if len(parts) > 4:
                clean_text = ", ".join(parts[:3] + [parts[-1]])  
            else:
                clean_text = full_text
            
            print(f"Extracted text - {clean_text}")
            patterns = ["%A, %B %d, %Y, %H:%M", "%A, %B %-d, %Y, %H:%M"]
            for pattern in patterns:
                try:
                    return datetime.strptime(clean_text, pattern)
                except ValueError:
                    continue
            
            print(f"Failed to parse datetime: {clean_text}")
            return None
        except Exception as e:
            print(f"Error extracting datetime on page {self.current_page}: {e}")
            return None

    def _scrape_table(self, cols=None):
        """
        Scrape the main table on a race card page.

        Parameters:
            cols (list, optional): List of column names to scrape. Defaults to all visible columns.

        Returns:
            pandas.DataFrame: DataFrame containing the table data.
        """
        try:
            print(f"Scraping data from page {self.current_page}...")
            table = Utils.wait_for_element(self.driver, By.ID, "racecardlist").find_element(By.TAG_NAME, "table")
            header_elements = table.find_element(By.TAG_NAME, "thead").find_element(By.TAG_NAME, "tr").find_elements(By.TAG_NAME, "td")

            headers = [
                header.text.strip() for header in header_elements
                if "display: none" not in header.get_attribute("style")
            ]

            headers = [header for header in headers if header in cols] if cols else headers
            rows = table.find_element(By.TAG_NAME, "tbody").find_elements(By.TAG_NAME, "tr")
            data = []

            for row in rows:
                row_data = {}
                tds = row.find_elements(By.TAG_NAME, "td")
                visible_tds = [
                    td for td in tds if "display: none" not in td.get_attribute("style")
                ]

                for header, td in zip(headers, visible_tds):
                    link_element = td.find_element(By.TAG_NAME, "a") if td.find_elements(By.TAG_NAME, "a") else None
                    row_data[header] = td.text.strip()
                    if link_element:
                        row_data[f"{header}_link"] = link_element.get_attribute("href")

                data.append(row_data)

            df = pd.DataFrame(data)
            print("Scraped DataFrame:")
            print(df)
            return df

        except Exception as e:
            print(f"Error occurred while scraping the table: {e}")
            return pd.DataFrame()

class BarrierTrialScraper(ASPXScraper):
    """
    HKJC Barrier-Trial scraper (work-in-progress).
    Mirrors the public API of ResultsScraper so downstream code can treat
    them interchangeably.
    """

    def __init__(self, driver):
        super().__init__(driver)
        self.base_url     = (
            "https://racing.hkjc.com/racing/information/"
            "English/Horse/BTResult.aspx"
        )
        self.current_date = None  
        self.track        = None   


    def navigate_to_date(self, date_value: str) -> bool:
        """
        Load the Barrier-Trial page for *date_value* (“dd/mm/yyyy”).

        Returns
        -------
        bool
            True  – page contains usable trials  
            False – HKJC shows the error banner, or validation fails
        """
        try:
            url_date = datetime.strptime(date_value, "%d/%m/%Y").strftime("%Y/%m/%d")
            target   = f"{self.base_url}?Date={url_date}"
            print(f"👉  {target}")
            self.driver.get(target)

            time.sleep(0.5)
            try:
                err = self.driver.find_element(By.ID, "errorContainer")
                if err.is_displayed():
                    print(f"↪  no barrier trials on {date_value} – skipping.")
                    return None
            except Exception:
                pass     

            if not Utils.wait_for_page_render(self.driver):
                print("⚠️  page did not finish rendering.")
                return False

            date_div = Utils.wait_for_element(
                self.driver,
                By.XPATH,
                "//div[contains(@class,'general_eng_text')"
                "      and contains(.,'Barrier Trial Date')]",
            )
            m = re.search(r"(\d{2}/\d{2}/\d{4})", date_div.text)
            if not m or m.group(1) != date_value:
                print(f"✗ date mismatch on {date_value}")
                return False

            hdr = Utils.wait_for_element(self.driver, By.CLASS_NAME, "btrcheader")
            hdr_up = hdr.text.strip().upper()

            self.track = (
                "ST" if "SHA TIN"      in hdr_up else
                "HV" if "HAPPY VALLEY" in hdr_up else
                "CH" if "CONGHUA"      in hdr_up else
                hdr_up                
            )

            self.current_date = date_value
            print(f"✅  {date_value} | Track = {self.track}")
            return True

        except Exception as e:
            print(f"❌  navigate_to_date({date_value}) failed: {e}")
            return False



    def _list_trial_anchors(self):
        """
        Returns the list of <a id="stbN"> WebElements that mark each trial.
        """
        return self.driver.find_elements(By.XPATH, "//a[starts-with(@id,'stb')]")


    def _parse_trial_header(self, anchor_elt):
        """
        Extract {batch, location, distance_m} from the subheader that follows
        the anchor.
        """
        try:
            td = anchor_elt.find_element(
                By.XPATH,
                "following-sibling::table[1]//td[contains(@class,'subheader')]",
            )
            header_text = " ".join(td.text.split())  
            m = re.match(r"Batch\s*(\d+)\s*-\s*(.*?)\s*-\s*(\d+)[mM]", header_text)
            if not m:
                print(f"⚠️ unexpected header format: {header_text}")
                return {}
            return {
                "batch":       int(m.group(1)),
                "location":    m.group(2).strip(),
                "distance_m":  int(m.group(3)),
            }
        except Exception as e:
            print(f"❌ _parse_trial_header failed: {e}")
            return {}

    @staticmethod
    def _normalize_url(url: str) -> str:
        """
        Force '/racing/information/English/' regardless of the original
        language segment.
        """
        return re.sub(
            r"(/racing/information/)([^/]+)(/)",
            r"\1English\3",
            url,
            flags=re.IGNORECASE,
        )

    def _scrape_trial_table(self, anchor_elt) -> pd.DataFrame:
        """
        Harvest the <table class='bigborder'> following *anchor_elt*.
        """
        try:
            table = anchor_elt.find_element(
                By.XPATH,
                "following-sibling::table[contains(@class,'bigborder')][1]",
            )

            headers = [
                " ".join(td.text.split())
                for td in table.find_elements(By.XPATH, ".//tr[1]/td")
            ]

            data = []
            for row in table.find_elements(By.XPATH, ".//tr[position()>1]"):
                cells = row.find_elements(By.TAG_NAME, "td")
                if not cells or not any(c.text.strip() for c in cells):
                    continue

                rec = {}
                for head, cell in zip(headers, cells):
                    txt = " ".join(cell.text.split()) or None
                    rec[head] = None if txt == "--" else txt

                    links = cell.find_elements(By.TAG_NAME, "a")
                    if links:
                        rec[f"{head}_link"] = self._normalize_url(
                            links[0].get_attribute("href")
                        )
                data.append(rec)

            return pd.DataFrame.from_records(data)

        except Exception as e:
            print(f"❌ _scrape_trial_table failed: {e}")
            return pd.DataFrame()


    def scrape_trial(self, anchor_elt) -> pd.DataFrame:
        """
        Combine header meta + performance rows for ONE heat.

        Adds:
            Date          – self.current_date  (string “dd/mm/yyyy”)
            trial_number  – batch number extracted from header
            horse_number  – 1,2,3,… in the order scraped

        Column order is fixed to:
            Date, trial_number, horse_number, <other columns…>

        Returns an empty DataFrame if either header or table fails.
        """
        meta = self._parse_trial_header(anchor_elt)
        if not meta:
            return pd.DataFrame()

        df = self._scrape_trial_table(anchor_elt)
        if df.empty:
            return pd.DataFrame()

        df["Date"]         = self.current_date
        df["trial_number"] = meta["batch"]
        df["horse_number"] = range(1, len(df) + 1)

        df["Track"]     = meta["location"]
        df["Dist."]   = meta["distance_m"]

        front = ["Date", "trial_number", "horse_number"]
        df = df[front + [c for c in df.columns if c not in front]]

        return df

    def scrape_trials_date(self, date_value: str) -> pd.DataFrame:
        """
        Navigate to *date_value* then aggregate every trial on that page.

        Returns a single DataFrame (empty if navigation or parsing fails).
        """
        if not self.navigate_to_date(date_value):
            return pd.DataFrame()

        anchors = self._list_trial_anchors()
        if not anchors:
            print(f"No trials found for {date_value}.")
            return pd.DataFrame()

        frames = []
        for a in anchors:
            df = self.scrape_trial(a)
            if not df.empty:
                frames.append(df)

        return (
            pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        )


    def scrape_trials_datelist(self, datelist):
        """
        Parameters
        ----------
        datelist : list[str]
            ["dd/mm/yyyy", …]

        Returns
        -------
        dict
            {
            "trial_performance_data": DataFrame,
            "invalid_dates":          list[str]
            }
        """
        all_data = []
        invalid  = []

        for d in datelist:
            print(f"▶  {d}")
            df = self.scrape_trials_date(d)
            if df.empty:
                invalid.append(d)
            else:
                all_data.append(df)
            time.sleep(1)       

        combined = (
            pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        )
        return {
            "trial_performance_data": combined,
            "invalid_dates": invalid,
        }


# if __name__ == '__main__':  

#     driver = webdriver.Chrome()
#     scraper = ResultsScraper(driver)
#     driver.get(scraper.base_url)
#     start_time = time.perf_counter()

#     years = [2017]
#     for year in years:
#         print(f"Starting scrape for year {year}...")
#         dates = generate_datetime(year)

#         result = scraper.scrape_results_all_pages_datelist(dates)

#         aspx_filename = f"aspx-results-{year}.csv"
#         result["race_performance_data"].to_csv(aspx_filename, index=False)

#         dividends_filename = f"dividends-{year}.json"

#         nested_dict_to_json(result["dividends_data"], dividends_filename)
#         to_json(result['invalid_pages'], f"invalid-pages-{year}.json")
    
#     end_time = time.perf_counter()

#     elapsed_time = end_time - start_time
#     print(f"Time taken to execute the function: {elapsed_time:.4f} seconds")

#     # HORSE SCRAPING

#     years = range(2010, 2025)
#     df_all = pd.DataFrame()

#     for year in years:
#         filename = os.path.join("../data/historical-data/aspx-results", f"aspx-results-{year}.csv")
#         df = pd.read_csv(filename)
#         df_all = pd.concat([df_all, df], ignore_index=True)

#     horse_links = df_all['Horse_link'].dropna().unique()
#     print(f"Found {len(horse_links)} unique horse links.")

#     driver = webdriver.Chrome()
#     horse_scraper = HorseScraper(driver)

#     dict_res = horse_scraper.scrape_season_records(horse_links)
    
#     season_records_filename = f"horses.json"
#     nested_dict_to_json(dict_res['season_records'], season_records_filename)

#     # NEXT RACECARD

#     driver = webdriver.Chrome()
#     racecard = RacecardScraper(driver)
#     driver.get(racecard.base_url)

#     races = racecard.scrape_next_race_card()

#     races.to_csv("../data/next-racecard/next-racecard.csv")

#     # BARRIER TRIALS
#     driver = webdriver.Chrome()
#     barrier_scraper = BarrierTrialScraper(driver)
#     driver.get(barrier_scraper.base_url)

#     years = range(2010, 2025)
#     for year in years: 
#         df = barrier_scraper.scrape_trials_datelist(generate_datetime(year))
#         df['trial_performance_data'].to_csv(f'../data/historical-data/barrier-trial-results/barrier-trial-results-{year}.csv'.format(year))

