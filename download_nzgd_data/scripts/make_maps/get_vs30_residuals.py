import pandas as pd
from pathlib import Path
from tqdm import tqdm
import natsort
from download_nzgd_data.lib import mapping



vs30_model_dir = Path("/home/arr65/data/nzgd/resources/vs30_map_resampled_from_Sung")
metadata_dir = Path("/home/arr65/data/nzgd/processed_data/cpt/metadata")
vs30_from_data_df = pd.read_csv(metadata_dir / "vs30_calculations.csv")
vs30_from_data_df.drop_duplicates(subset=["record_name"], keep='first', inplace=True)
output_file = metadata_dir / "closest_grid_point_in_vs30_model.csv"

num_procs = 8

### Setting the station_id to be the record_name for compatibility with functions that use record_name
model_vs30_ll_df = pd.read_csv(vs30_model_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.ll",
                                sep=" ",
                                header=None,
                                names=["model_longitude",
                                       "model_latitude",
                                       "model_grid_point_name"])
model_vs30_value_df = pd.read_csv(vs30_model_dir/"non_uniform_whole_nz_with_real_stations-hh400_v20p3_land.vs30",
                                   sep=" ", header=None, names=["model_grid_point_name", "model_vs30"])

unmatched_vs30_model_df = model_vs30_value_df.merge(model_vs30_ll_df, on=["model_grid_point_name"], how="left")
#vs30_from_data_df = vs30_from_data_df.iloc[:1000]

print("finding closest model grid point for each record...")

closest_model_grid_point_df = mapping.calc_all_closest_cpt_dist(
    lon_lat_to_consider_df=vs30_from_data_df[["record_name","longitude", "latitude"]],
    all_lon_lat_df=unmatched_vs30_model_df[["model_grid_point_name", "model_longitude", "model_latitude"]],
    n_procs=num_procs)

vs30_from_model_df = closest_model_grid_point_df.merge(unmatched_vs30_model_df, left_on="closest_model_grid_point_name",
                                                              right_on="model_grid_point_name", how="left",
                                                              suffixes=(None, "_model"))

vs30_from_data_and_model_df = vs30_from_data_df.merge(vs30_from_model_df,
                                                      left_on="record_name",
                                                      right_on="record_name_from_data", how="left",
                                                      suffixes=(None, "_model"))
vs30_from_data_and_model_df.to_csv(metadata_dir / "vs30_from_data_and_model.csv")
