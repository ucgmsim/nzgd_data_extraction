import pandas as pd

location_match_df = pd.read_csv("/home/arr65/data/nzgd/stats_plots/sung_id_with_nzgd_match.csv")

matching_names_idx = location_match_df["cpt_name"] == location_match_df["closest_nzgd_cpt_id"]

names_match_df = location_match_df[matching_names_idx]

median_dist_matching_names = names_match_df["closest_dist"].median()


names_not_match_df = location_match_df[~matching_names_idx]
median_dist_not_matching_names = names_not_match_df["closest_dist"].median()

print(matching_names_idx)
print()