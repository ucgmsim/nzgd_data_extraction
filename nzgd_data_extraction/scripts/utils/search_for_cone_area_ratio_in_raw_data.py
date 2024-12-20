import pandas as pd
from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
from nzgd_data_extraction.lib.processing_helpers import find_encoding
import multiprocessing as mp



def search_cpt_record_for_string(cpt_dir):
    xlsx_files = list(cpt_dir.glob("*.xlsx"))
    xls_files = list(cpt_dir.glob("*.xls"))
    csv_files = list(cpt_dir.glob("*.csv"))
    txt_files = list(cpt_dir.glob("*.txt"))
    ags_files = list(cpt_dir.glob("*.ags"))

    files = xlsx_files + xls_files + csv_files + txt_files + ags_files

    if len(files) == 0:
        return None

    for file_path in files:

        if file_path.suffix in [".xlsx", ".xls"]:

            try:

                sheet_names = pd.ExcelFile(file_path).sheet_names

                for sheet_name in sheet_names:

                    df = pd.read_excel(
                        file_path,
                        sheet_name=sheet_name,
                        header=None,
                        parse_dates=False,
                    )

                    if df.map(lambda x: isinstance(x, str) and search_string in x.lower()).any().any():
                        return str(file_path)

            except Exception as e:
                # print(f"Error reading {file_path}: {e}")
                # continue
                return None

        else:

            ### Identify the codec of the file by trying to read it with
            encoding = find_encoding(file_path)

            try:
                with open(file_path, 'r', encoding=encoding) as file:
                    file_content = file.read()
                    if search_string in file_content:
                        return str(file_path)

            except Exception as e:
                # print(f"Error reading {file_path}: {e}")
                # continue
                return None


if __name__ == "__main__":
    cpt_list = natsorted(
        list(Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd/cpt").glob("*")))

    #cpt_list = cpt_list[:1000]

    search_string = "area ratio"


    results = []
    num_workers = 8
    with mp.Pool(processes=num_workers) as pool:
        results.extend(
            list(
                tqdm(
                    pool.imap(search_cpt_record_for_string, cpt_list),
                    total=len(cpt_list),
                )
            )
        )



    results = [result for result in results if result is not None]

    # Write the list of strings to the file
    with open("/home/arr65/data/nzgd/resources/example_spreadsheets/cpt_records_with_area_ratio.txt", 'w') as file:
        for line in results:
            file.write(line + '\n')

    print()

