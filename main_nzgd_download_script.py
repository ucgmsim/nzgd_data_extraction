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

url_df = pd.read_csv('/home/arr65/data/nzgd/nzgd_export.csv')
url_df = url_df[url_df["TypeofInvestigation"] == 'CPT']
### url_df = url_df.iloc[:2]

print(f"{len(url_df)} CPT records to download")

# Load environment variables from .env_nzgd file
load_dotenv(".env_nzgd")

# Define the login URL and the list of URLs to scrape
login_url = 'https://www.nzgd.org.nz/Registration/Login.aspx'
data_urls = url_df['PopUpLink'].tolist()

# Get login credentials from environment variables
username_str = os.getenv('NZGD_USERNAME')
password_str = os.getenv('NZGD_PASSWORD')

# Set up the download directory
download_dir = Path("/home/arr65/data/nzgd/downloaded_files/unorganized_downloads")
os.makedirs(download_dir, exist_ok=True)

# Set up the state directory
state_dir = download_dir.parent.parent / "last_processed_states"
os.makedirs(state_dir, exist_ok=True)

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


def wait_for_page_load(driver):
    WebDriverWait(driver, 10).until(
        lambda d: d.execute_script('return document.readyState') == 'complete'
    )


# Function to process a chunk of URLs
def process_chunk(chunk_index, data_url_chunk):
    state_file = state_dir / f'last_processed_state_chunk_{chunk_index}.txt'
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            last_processed_index = int(f.read().strip())
    else:
        last_processed_index = 0

    while last_processed_index < len(data_url_chunk):
        try:
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
            for data_url_index in range(last_processed_index, len(data_url_chunk)):
                data_url = data_url_chunk[data_url_index]
                start_time_for_this_url = time.time()
                print(f"Downloading from {data_url} (Chunk {chunk_index + 1}, URL {data_url_index + 1}/{len(data_url_chunk)})")

                # Save the current index to file
                with open(state_file, 'w') as f:
                    f.write(str(data_url_index))

                # Navigate to the data URL
                driver.get(data_url)
                #wait.until(EC.presence_of_element_located((By.TAG_NAME, 'a')))  # Wait for the page to load
                #wait_for_page_load(driver)
                time.sleep(1)
                # Parse the HTML content using BeautifulSoup
                soup = BeautifulSoup(driver.page_source, 'html.parser')

                # Find and click the link
                document_links = soup.find_all('a', href=True)
                print()
                for link in document_links:
                    link_text = link.text.strip()
                    if link_text.endswith(('.txt', '.xls', '.ags', '.pdf')):
                        print(f'Clicking link: {link_text}')
                        element = wait.until(EC.element_to_be_clickable((By.LINK_TEXT, link_text)))
                        element.click()
                        #wait.until(EC.staleness_of(element))  # Wait for the download to complete
                        time.sleep(1)  # Wait for the download to complete

                end_time_for_this_url = time.time()
                print(f"Time taken for this URL: {end_time_for_this_url - start_time_for_this_url:.2f} seconds")

            driver.quit()
            break

        except Exception as e:
            print(f"Error processing {data_url_chunk[last_processed_index]}: {e}")
            print(f"Exception type: {type(e).__name__}")
            print(f"Exception args: {e.args}")
            driver.quit()
            time.sleep(10)  # Wait before retrying

# Define the number of chunks
number_of_chunks = 8  # Adjust the number of chunks as needed

# Divide data_urls into chunks
data_url_chunks = chunkify(data_urls, number_of_chunks)

# Use multiprocessing to process chunks in parallel
if __name__ == '__main__':
    with Pool(processes=number_of_chunks) as pool:
        pool.starmap(process_chunk, enumerate(data_url_chunks))

# Organize downloaded files into folders based on their file types
for file in download_dir.iterdir():
    if file.is_file():
        file_extension = file.suffix[1:]  # Get file extension without the dot
        destination_dir = download_dir.parent / "organized_downloads" / file_extension
        os.makedirs(destination_dir, exist_ok=True)
        shutil.copy(str(file), str(destination_dir / file.name))

end_time = time.time()
print(f"Time taken: {(end_time - start_time)/3600:.2f} hours")

# delete the state files as they are not needed at this point
for state_file in state_dir.iterdir():
    state_file.unlink()

