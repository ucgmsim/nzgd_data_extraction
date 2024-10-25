"""
This script compresses the NZGD directory structure at the city level.
"""

from pathlib import Path
from download_nzgd_data.lib import organise
import multiprocessing
import shutil

#hypocentre_mirror_dir = Path("/home/arr65/data/nzgd/hypocentre_mirror_copy")
dropbox_mirror_dir = Path("/home/arr65/data/nzgd/dropbox_mirror")

## use copytree to copy the hypocentre_mirror to the dropbox_mirror
## this will take a few minutes
#shutil.copytree(hypocentre_mirror_dir, dropbox_mirror_dir)

## Delete index.html and date_of_last_nzgd_retrieval.txt
# files_to_delete = [path for path in (dropbox_mirror_dir/"nzgd").glob("*") if path.is_file()]
# for file in files_to_delete:
#     file.unlink()

all_files = [file for file in list(dropbox_mirror_dir.rglob('*')) if file.is_file()]

non_tarxz_files = [file for file in all_files if file.suffix != ".xz"]

## number of directory layers to keep
dir_structure_depth = 11
limited_paths = [Path(*path.parts[:dir_structure_depth]) for path in non_tarxz_files]

unique_paths = sorted(list(set(limited_paths)))
print()

with multiprocessing.Pool(processes=6) as pool:
    pool.map(organise.replace_folder_with_tar_xz, unique_paths)


