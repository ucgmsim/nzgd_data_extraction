import pandas as pd
import glob

file_list = glob.glob("/home/arr65/data/nzgd/downloads_and_metadata/analysis_ready_data/cpt/data/*.parquet")

### Each parquet file contains the following columns:
#'record_name',
# 'latitude',
# 'longitude',
# 'depth_m',
# 'cone_resistance_qc_mpa',
# 'sleeve_friction_fs_mpa',
# 'pore_pressure_u2_mpa'

### Method 1: Load each parquet file as a pandas dataframe. This dataframe includes metadata that
### can be accessed using the .attrs attribute

single_cpt_df = pd.read_parquet(file_list[0])
# can access the original file name using:
original_file_name = single_cpt_df.attrs["original_file_name"]
nzgd_meta_data = single_cpt_df.attrs["nzgd_meta_data"]

# ### Method 2: Load all parquet files into a single dataframe and then filter using the record_name, latitude, or longitude columns
# concatenated_data_frames = pd.read_parquet(file_list)
#
# # filter a specific CPT record like this:
# cpt_1 = concatenated_data_frames[concatenated_data_frames["record_name"] == "CPT_1"]

print()



