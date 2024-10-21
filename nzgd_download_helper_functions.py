"""
Functions to facilitate automated downloading of data from the New Zealand Geotechnical Database (NZGD).
"""

import os
import time
from pathlib import Path
import numpy as np

import pandas as pd
import toml
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

import config as cfg

config = cfg.Config()


# Function to set up Selenium WebDriver
def setup_driver(download_dir):
    """
    Sets up the Selenium WebDriver with specified options and preferences.

    Parameters
    ----------
    download_dir : Path
        The directory where downloaded files will be saved.

    Returns
    -------
    selenium.webdriver.Chrome
        An instance of the Chrome WebDriver configured with the specified options.
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.binary_location = "/usr/bin/google-chrome"  # Path to Chrome binary
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True,
    }
    chrome_options.add_experimental_option("prefs", prefs)

    # Specify the path to the manually downloaded ChromeDriver
    driver_path = "/home/arr65/.wdm/drivers/chromedriver/linux64/128.0.6613.137/chromedriver"
    if not os.path.exists(driver_path):
        driver_path = ChromeDriverManager().install()

    return webdriver.Chrome(
        service=Service(driver_path), options=chrome_options
    )


def wait_for_page_load(driver):
    """
    Waits for the page to fully load by checking the document.readyState.

    Parameters
    ----------
    driver : selenium.webdriver.Chrome
        The Selenium WebDriver instance used to interact with the web page.
    """
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


# Function to process a chunk of URLs
def process_df_row(url_df_row_index, url_df_row):
    """
    Processes a row from a DataFrame containing URLs, logs into the NZGD website,
    navigates to the data URL, and downloads the files linked on the page.

    Parameters
    ----------
    url_df_row_index : int
        The index of the row in the DataFrame.
    url_df_row : pandas.Series
        A row from the DataFrame containing the URL and ID.

    Returns
    -------
    None
    """
    login_url = config.get_value("login_url")
    username_str = os.getenv("NZGD_USERNAME")
    password_str = os.getenv("NZGD_PASSWORD")
    load_wait_time_s = config.get_value("load_wait_time_s")

    download_dir = Path(config.get_value("high_level_download_dir")) / url_df_row["ID"]
    os.makedirs(download_dir, exist_ok=True)
    driver = setup_driver(download_dir)

    # Log in to the website
    driver.get(login_url)

    # Wait for the username field to be present
    wait = WebDriverWait(driver, load_wait_time_s)
    username = wait.until(
        EC.presence_of_element_located(
            (By.NAME, "ctl00$MainContent$LoginControl$LoginBox$UserName")
        )
    )
    password = driver.find_element(
        By.NAME, "ctl00$MainContent$LoginControl$LoginBox$Password"
    )

    # Enter login credentials
    username.send_keys(username_str)
    password.send_keys(password_str)
    password.send_keys(Keys.RETURN)

    # Wait for specific text that indicates a successful login
    wait.until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Home')]"))
    )  # Replace 'Welcome' with the actual text

    data_url = url_df_row["URL"]

    # Navigate to the data URL
    driver.get(data_url)
    time.sleep(load_wait_time_s)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(driver.page_source, "html.parser")

    document_links = soup.find_all("a", href=True)

    file_links = [link for link in document_links if "." in link.text.strip()]
    file_links_text = [link.text.strip() for link in file_links]

    link_as_str_dict = {url_df_row["ID"] : []}
    file_names_dict = {url_df_row["ID"] : file_links_text}

    for link in file_links:

        link_as_str_dict[url_df_row["ID"]].append(str(link))
        element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, link.text)))
        element.click()

        time.sleep(load_wait_time_s)  # Wait for the download to complete
    print()
    np.savetxt(Path(config.get_value("downloaded_record_note_per_record")) / f"{url_df_row_index}.txt", np.array([url_df_row_index]))
    with open(Path(config.get_value("name_to_files_dir_per_record")) / f"{url_df_row_index}.toml", "w") as toml_file:
        toml.dump(file_names_dict, toml_file)

    with open(Path(config.get_value("name_to_link_str_dir_per_record")) / f"{url_df_row_index}.toml", "w") as toml_file:
        toml.dump(link_as_str_dict, toml_file)

    driver.quit()


# Function to process a chunk of URLs
def get_metadata_from_nzgd_record_page(url_df_row_index, url_df_row):
    """
    Processes a row from a DataFrame containing URLs, logs into the NZGD website,
    navigates to the data URL, and downloads the files linked on the page.

    Parameters
    ----------
    url_df_row_index : int
        The index of the row in the DataFrame.
    url_df_row : pandas.Series
        A row from the DataFrame containing the URL and ID.

    Returns
    -------
    None
    """
    login_url = config.get_value("login_url")
    username_str = os.getenv("NZGD_USERNAME")
    password_str = os.getenv("NZGD_PASSWORD")
    load_wait_time_s = config.get_value("load_wait_time_s")

    driver = setup_driver(config.get_value("name_to_metadata_dir_per_record"))

    # Log in to the website
    driver.get(login_url)

    # Wait for the username field to be present
    wait = WebDriverWait(driver, load_wait_time_s)
    username = wait.until(
        EC.presence_of_element_located(
            (By.NAME, "ctl00$MainContent$LoginControl$LoginBox$UserName")
        )
    )
    password = driver.find_element(
        By.NAME, "ctl00$MainContent$LoginControl$LoginBox$Password"
    )

    # Enter login credentials
    username.send_keys(username_str)
    password.send_keys(password_str)
    password.send_keys(Keys.RETURN)

    # Wait for specific text that indicates a successful login
    wait.until(
        EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Home')]"))
    )  # Replace 'Welcome' with the actual text

    data_url = url_df_row["URL"]

    # Navigate to the data URL
    driver.get(data_url)
    time.sleep(load_wait_time_s)

    nzgd_id = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//h4[contains(text(), 'NZGD ID:')]"))
    ).text

    # Extract Project Name (optional)
    project_name = driver.find_element(By.NAME, "projectName").get_attribute("value")

    # Extract Project Client
    project_client = driver.find_element(By.NAME, "projectClient").get_attribute("value")

    # Extract Supervising Company
    supervising_company = driver.find_element(By.NAME, "engineeringCompany").get_attribute("value")

    # Extract Status
    status = driver.find_element(By.XPATH, "//label[text()='Status :']/following-sibling::div/label").text

    # Extract Investigation Type
    investigation_type = driver.find_element(By.NAME, "investigationType").get_attribute("value")

    # Extract Location ID
    location_id = driver.find_element(By.NAME, "reference").get_attribute("value")

    # Extract Supervising Technician
    technician = driver.find_element(By.NAME, "technician").get_attribute("value")

    # Extract Drill Rig Name
    drill_rig = driver.find_element(By.NAME, "drillRig").get_attribute("value")

    # Extract Drilling Company
    drilling_company = driver.find_element(By.NAME, "drillingCompany").get_attribute("value")

    # Extract End Date
    end_date = driver.find_element(By.NAME, "dateInvestigation").get_attribute("value")

    # Extract total depth
    total_depth = driver.find_element(By.NAME, "depthOfSoilTest").get_attribute("value")

    # Extract Pre-drill Depth
    pre_drill_depth = driver.find_element(By.NAME, "preDrillDepth").get_attribute("value")

    # Extract Known Issue (Yes/No Checkboxes)
    known_issue_no = driver.find_element(By.XPATH,
                                         "//input[@name='isKnownIssueWithRawData'][@value='false']").get_attribute(
        "checked")
    known_issue_yes = driver.find_element(By.XPATH,
                                          "//input[@name='isKnownIssueWithRawData'][@value='true']").get_attribute(
        "checked")

    # Extract Ground Water Measured (Yes/No Checkboxes)
    ground_water_no = driver.find_element(By.XPATH, "//input[@name='isGwMeasured'][@value='false']").get_attribute(
        "checked")
    ground_water_yes = driver.find_element(By.XPATH, "//input[@name='isGwMeasured'][@value='true']").get_attribute(
        "checked")

    # Extract Cone Serial Number
    cone_serial_number = driver.find_element(By.NAME, "coneId").get_attribute("value")

    # Extract Remarks
    remarks = driver.find_element(By.NAME, "remark").get_attribute("value")


    data = {
        'nzgd_id': nzgd_id.strip("NZGD ID:"),
        'technician': technician.strip("string:"),
        'drill_rig': drill_rig.strip("string:"),
        'drilling_company': drilling_company.strip("string:"),
        'end_date': end_date,
        'project_name': project_name,
        'project_client': project_client,
        'supervising_company': supervising_company.strip("string:"),
        'status': status,
        'investigation_type': investigation_type.strip("string:"),
        'location_id': location_id,
        'total_depth': total_depth,
        'pre_drill_depth': pre_drill_depth,
        'known_issue_no': known_issue_no,
        'known_issue_yes': known_issue_yes,
        'ground_water_no': ground_water_no,
        'ground_water_yes': ground_water_yes,
        'cone_serial_number': cone_serial_number,
        'remarks': remarks
    }

    # Create a DataFrame with a single row from the extracted data
    metadata_df = pd.DataFrame([data])

    # Save the metadata to a CSV file
    metadata_df.to_csv(Path(config.get_value("name_to_metadata_dir_per_record")) / f"row_{url_df_row_index}_metadata.csv", index=False)
    #
    #
    #
    # Quit the driver
    driver.quit()