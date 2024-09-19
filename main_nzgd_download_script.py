import os
import time
import shutil
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from dotenv import load_dotenv
from webdriver_manager.chrome import ChromeDriverManager
from pathlib import Path
from multiprocessing import Pool

start_time = time.time()


# Function to set up Selenium WebDriver
def setup_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.binary_location = "/usr/bin/google-chrome"  # Path to Chrome binary
    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }
    chrome_options.add_experimental_option("prefs", prefs)
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# Function to divide data_urls into chunks
def chunkify(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


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
    return [df.iloc[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


# Save chunks to CSV using pandas
def save_chunks_to_csv(chunks, output_file):
    max_length = max(len(chunk) for chunk in chunks)
    chunk_dict = {f"chunk_{i+1}": chunk + [''] * (max_length - len(chunk)) for i, chunk in enumerate(chunks)}
    df = pd.DataFrame(chunk_dict)
    df.to_csv(output_file, index=False)

def wait_for_page_load(driver):
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script('return document.readyState') == 'complete'
    )


def get_downloaded_cpt_nums(extension):

    """
    Get the numbers of the CPT files that have been downloaded

    Parameters
    ----------
    extension : str
        The extension of the file type to check for

    Returns
    -------
    set
        A set containing the numbers of the downloaded CPT files
    """

    downloaded_files = set(os.listdir(Path("/home/arr65/data/nzgd/downloaded_files/organized_downloads") / extension))

    downloaded_cpt_nums = set()

    for file in downloaded_files:
        if file.startswith("CPT"):

            if len(file.split("_")) > 1:

                cpt_num = file.split("_")[1]

                downloaded_cpt_nums.add(f"CPT_{cpt_num}")

    return downloaded_cpt_nums

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

    all_links = soup.find_all('a', href=True)

    available_files = [link.text.strip() for link in all_links if "." in link.text.strip()]

    return len(available_files)


# Function to process a chunk of URLs
def process_chunk(chunk_index, url_chunk_df):

    """
    mode = 'download' or 'get_file_names'
    """

    downloaded_xls_cpt = get_downloaded_cpt_nums("xls")
    downloaded_pdf_cpt = get_downloaded_cpt_nums("pdf")
    downloaded_ags_cpt = get_downloaded_cpt_nums("ags")
    downloaded_txt_cpt = get_downloaded_cpt_nums("txt")


    load_wait_time_s = 2

    state_file = state_dir / f'last_processed_state_chunk_{chunk_index}.txt'
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_processed_index = int(f.read().strip())
    else:
        last_processed_index = 0

    file_names_dict = {}

    driver = setup_driver()

    # Log in to the website
    driver.get(login_url)

    # Wait for the username field to be present
    wait = WebDriverWait(driver, 10)
    username = wait.until(EC.presence_of_element_located((By.NAME, 'ctl00$MainContent$LoginControl$LoginBox$UserName')))
    password = driver.find_element(By.NAME, 'ctl00$MainContent$LoginControl$LoginBox$Password')

    # Enter login credentials
    username.send_keys(username_str)
    password.send_keys(password_str)
    password.send_keys(Keys.RETURN)

    # Wait for specific text that indicates a successful login
    wait.until(EC.presence_of_element_located((By.XPATH, "//*[contains(text(), 'Home')]")))  # Replace 'Welcome' with the actual text
    print("Login successful")
    print()

    # Loop through each URL in the chunk starting from the last processed index
    for data_url_index in range(last_processed_index, len(url_chunk_df)):

        name = url_chunk_df.at[data_url_index, 'Name']

        ## See if there are already downloaded files of all possible types

        #### This is problematic because it seems like a few records have an uncommon file type like .csv
        # if ((name in downloaded_xls_cpt) and (name in downloaded_pdf_cpt) and (name in downloaded_ags_cpt) and
        #         (name in downloaded_txt_cpt)):
        #     print(f"Skipping {name} as all possible files have already been downloaded")
        #     continue

        data_url = url_chunk_df.at[data_url_index, 'PopUpLink']

        file_names_dict[data_url] = []

        start_time_for_this_url = time.time()
        print(f"Downloading from {data_url} (Chunk {chunk_index + 1}, URL {data_url_index + 1}/{len(url_chunk_df)})")

        # Save the current index to file
        with open(state_file, 'w') as f:
            f.write(str(data_url_index))

        # Navigate to the data URL
        driver.get(data_url)
        #wait.until(EC.presence_of_element_located((By.TAG_NAME, 'a')))  # Wait for the page to load
        #wait_for_page_load(driver)
        print(f"waiting {load_wait_time_s} seconds for page to load")
        time.sleep(load_wait_time_s)

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        if get_number_of_available_files(soup) < 3:
            ## Wait a bit longer in case the page has not finished loading
            print(f"waiting {load_wait_time_s} seconds to make sure page has loaded")
            time.sleep(load_wait_time_s)

        # Find and click the link
        document_links = soup.find_all('a', href=True)

        file_links = [link for link in document_links if "." in link.text.strip()]
        file_names_dict[data_url].extend([link.text.strip() for link in file_links])

        for link in file_links:

            if 'xls' in link.text:
                if name in downloaded_xls_cpt:
                    print(f"Skipping {link.text} as it has already been downloaded")
                    continue
            elif 'pdf' in link.text:
                if name in downloaded_pdf_cpt:
                    print(f"Skipping {link.text} as it has already been downloaded")
                    continue
            elif 'ags' in link.text:
                if name in downloaded_ags_cpt:
                    print(f"Skipping {link.text} as it has already been downloaded")
                    continue
            elif 'txt' in link.text:
                if name in downloaded_txt_cpt:
                    print(f"Skipping {link.text} as it has already been downloaded")
                    continue

            print(f'Clicking link: {link.text}')
            element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, link.text)))
            element.click()
            #wait.until(EC.staleness_of(element))  # Wait for the download to complete

            print(f"waiting {load_wait_time_s} seconds for download to complete")
            time.sleep(load_wait_time_s)  # Wait for the download to complete

        end_time_for_this_url = time.time()
        print(f"Time taken for this URL: {end_time_for_this_url - start_time_for_this_url:.2f} seconds")

    driver.quit()

        # except Exception as e:
        #     print(f"Error processing {data_url_chunk[last_processed_index]}: {e}")
        #     print(f"Exception type: {type(e).__name__}")
        #     print(f"Exception args: {e.args}")
        #     driver.quit()
        #     time.sleep(10)  # Wait before retrying




if __name__ == '__main__':

#    url_df = pd.read_csv('/home/arr65/data/nzgd/nzgd_index_files/csv_files/small_qgis_export_index_18_sep_2024.csv')
    url_df = pd.read_csv('/home/arr65/data/nzgd/nzgd_index_files/csv_files/qgis_export_index_18_sep_2024.csv')
    ### url_df = url_df.iloc[:2]
    url_df = url_df[url_df["TypeofInvestigation"] == "CPT"][["OBJECTID", "Name", "PopUpLink"]]
    #url_df = url_df[["OBJECTID", "Name", "PopUpLink"]]

    # Load environment variables from .env_nzgd file
    load_dotenv(".env_nzgd")

    # Define the login URL and the list of URLs to scrape
    login_url = 'https://www.nzgd.org.nz/Registration/Login.aspx'

    # Get login credentials from environment variables
    username_str = os.getenv('NZGD_USERNAME')
    password_str = os.getenv('NZGD_PASSWORD')

    # Set up the download directory
    download_dir = Path("/home/arr65/data/nzgd/downloaded_files/unorganized_downloads")
    os.makedirs(download_dir, exist_ok=True)

    # Set up the state directory
    state_dir = download_dir.parent.parent / "last_processed_states"
    os.makedirs(state_dir, exist_ok=True)

    # Define the number of chunks
    number_of_chunks = 6  # Adjust the number of chunks as needed

    # Divide data_urls into chunks
    data_url_chunks = chunkify_dataframe(url_df, number_of_chunks)

    # Use multiprocessing to process chunks in parallel
    with Pool(processes=number_of_chunks) as pool:
        pool.starmap(process_chunk, enumerate(data_url_chunks))

    end_time = time.time()
    print(f"Time taken: {(end_time - start_time)/3600:.2f} hours")

    # # delete the state files as they are not needed at this point
    # for state_file in state_dir.iterdir():
    #     state_file.unlink()

