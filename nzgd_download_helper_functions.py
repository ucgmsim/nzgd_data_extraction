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


# Function to divide data_urls into chunks
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def chunkify_dataframe(df, n):
    """
    Divide a pandas DataFrame into n chunks.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be divided.
    n : int
        The number of chunks to divide the DataFrame into.

    Returns
    -------
    list of pandas.DataFrame
        A list containing the DataFrame chunks.
    """
    k, m = divmod(len(df), n)
    return [df.iloc[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


# Save chunks to CSV using pandas
def save_chunks_to_csv(chunks, output_file):
    max_length = max(len(chunk) for chunk in chunks)
    chunk_dict = {
        f"chunk_{i+1}": chunk + [""] * (max_length - len(chunk))
        for i, chunk in enumerate(chunks)
    }
    df = pd.DataFrame(chunk_dict)
    df.to_csv(output_file, index=False)


def wait_for_page_load(driver):
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script("return document.readyState") == "complete"
    )


def get_number_of_available_files(soup):
    """
    Get the number of available files from the soup object

    Parameters
    ----------
    soup : BeautifulSoup
        The BeautifulSoup object containing the HTML content

    Returns
    -------
    int
        The number of available files
    """

    all_links = soup.find_all("a", href=True)

    available_files = [
        link.text.strip() for link in all_links if "." in link.text.strip()
    ]

    return len(available_files)


# Function to process a chunk of URLs
def process_chunk(url_df_row_index, url_df_row):



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

    np.savetxt(Path(config.get_value("downloaded_record_note_per_record")) / f"{url_df_row_index}.txt", np.array([url_df_row_index]))

    with open(Path(config.get_value("name_to_files_dir_per_record")) / f"{url_df_row_index}.toml", "w") as toml_file:
        toml.dump(file_names_dict, toml_file)

    with open(Path(config.get_value("name_to_link_str_dir_per_record")) / f"{url_df_row_index}.toml", "w") as toml_file:
        toml.dump(link_as_str_dict, toml_file)

    driver.quit()
