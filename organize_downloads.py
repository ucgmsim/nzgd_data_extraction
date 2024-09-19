import os
import shutil
from pathlib import Path

organized_base_dir = Path("/home/arr65/data/nzgd/downloaded_files/organized_downloads")

download_part_nums = list(range(12,14))

for download_part_num in download_part_nums:

    print(f"Processing download part {download_part_num}")

    unorganized_dir = Path(f"/home/arr65/data/nzgd/downloaded_files/unorganized_downloads_part{download_part_num}")

    # Organize downloaded files into folders based on their file types
    for fileidx, file in enumerate(unorganized_dir.iterdir()):
        if file.is_file():

            if "(" in file.name:
                continue

            file_extension = file.suffix[1:]  # Get file extension without the dot
            destination_dir = organized_base_dir / file_extension
            os.makedirs(destination_dir, exist_ok=True)

            if fileidx % 100 == 0:
                print(f"Processing download part {download_part_num} file {fileidx + 1} of {len(list(unorganized_dir.iterdir()))}")

                print(f"Copying {file} to {destination_dir / file.name}")


            shutil.copy(str(file), str(destination_dir / file.name))


### Delete files with "(" in their names (Chrome adds (X) to the end of files with the same name)
for organized_dir in organized_base_dir.iterdir():

    if organized_dir.is_dir():

        for file in organized_dir.iterdir():

            if file.is_file():

                if "(" in file.name:
                    print(f"Deleting {file}")
                    file.unlink()