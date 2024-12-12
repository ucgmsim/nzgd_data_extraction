import pandas as pd
import numpy as np
from pathlib import Path

from tqdm import tqdm

from plotly.subplots import make_subplots
import plotly.graph_objects as go




record_name = "CPT_1"
# cpt_df = pd.read_parquet("/home/arr65/src/nzgd_map_from_webplate/instance/extracted_cpt_and_scpt_data.parquet",
#                              filters=[("record_name", "==", record_name)]).reset_index()

instance_path = Path("/home/arr65/src/nzgd_map_from_webplate/instance")

vs30_df_all_records = pd.read_parquet(
    instance_path / "website_database.parquet"
).reset_index()

record_details_df = vs30_df_all_records[
    vs30_df_all_records["record_name"] == record_name
    ]

max_depth_for_record = record_details_df["max_depth"].unique()[0]

if max_depth_for_record < 5:
    vs30_correlation_explanation_text = (f"Unable to estimate a Vs30 value for {record_name} as it has a maximum depth "
                        f"of {max_depth_for_record} m, while depths of at least 10 m and 5 m are required for "
                        "the Boore et al. (2004) and Boore et al. (2011) correlations, respectively.")
    show_vs30_values = False

elif 5 <= max_depth_for_record < 10:
    vs30_correlation_explanation_text = (f"{record_name} has a maximum depth of {max_depth_for_record} m so only the Boore et al. (2011) "
                        "Vs30 correlation can be used as it requires a depth of at least 5 m, while the "
                        "Boore et al. (2004) correlation requires a depth of at least 10 m.")
    show_vs30_values = True
    record_details_df = record_details_df[record_details_df["vs30_correlation"] == "boore_2011"]

else:
    vs30_correlation_explanation_text = (f"{record_name} has a maximum depth of {max_depth_for_record} m so both the Boore et al. (2004) "
                        "and Boore et al. (2011) Vs30 correlations can be used, as they require depths of at least "
                        "10 m and 5 m, respectively.")
    show_vs30_values = True

## Make a new column for the Vs30 value to display in the hover text
database_df = vs30_df_all_records
## Make a new column for the Vs30 value to display in the hover text

vs30_correlation = "boore_2011"

database_df["Vs30"] =  database_df["vs30"]

if vs30_correlation == "boore_2011":
    reason_text = "Unable to estimate as Boore et al. (2011) Vs to Vs30 correlation requires a depth of at least 5 m"
    min_required_depth = 5
else:
    reason_text = "Unable to estimate as Boore et al. (2004) Vs to Vs30 correlation requires a depth of at least 10 m"
    min_required_depth = 10


# Vectorized operations for better performance
database_df.loc[database_df["max_depth"] < min_required_depth, "Vs30"] = reason_text
database_df.loc[(database_df["max_depth"] >= min_required_depth) & (np.isnan(database_df["vs30"]) | (database_df["vs30"] == 0)), "Vs30"] = "Vs30 calculation failed even though CPT depth is sufficient"
database_df.loc[(database_df["max_depth"] >= min_required_depth) & ~(np.isnan(database_df["vs30"]) | (database_df["vs30"] == 0)), "Vs30"] = database_df["vs30"].apply(lambda x: f"{x:.2f}")

# for row_index in range(len(database_df)):
#
#     if database_df.loc[row_index,"max_depth"] < min_required_depth:
#         database_df.loc[row_index,"Vs30"] = reason_text
#     elif ((np.isnan(database_df.loc[row_index,"vs30"]) | (database_df.loc[row_index,"vs30"] == 0))):
#         database_df.loc[row_index,"Vs30"] = "Vs30 calculation failed even though CPT depth is sufficient"
#     else:
#         database_df.loc[row_index,"Vs30"] = f"{database_df.loc[row_index,'vs30']:.2f}"
#




print()

# if vs30_correlation == "boore_2011":
#     ## Need depth at least 5 m
#     database_df["Vs30"] = database_df["vs30"].apply(
#         lambda x: "Unable to estimate as Boore et al. (2011) Vs to Vs30 correlation requires a depth of at least 5 m"
#         if database_df["max_depth"] < 5 else f"{x:.2f}")
#
# if vs30_correlation == "boore_2004":
#     ## Need depth at least 10 m
#     database_df["Vs30"] = database_df["vs30"].apply(
#         lambda x: "Unable to estimate as Boore et al. (2004) Vs to Vs30 correlation requires a depth of at least 5 m"
#         if database_df["max_depth"] < 10 else f"{x:.2f}")
#     pass
#
# database_df["Vs30"] = database_df["Vs30"].apply(lambda x: "Vs30 calculation failed even though CPT depth is sufficient"
# if pd.isna(x) or x == 0 else f"{x:.2f}")
#
# print()
#
