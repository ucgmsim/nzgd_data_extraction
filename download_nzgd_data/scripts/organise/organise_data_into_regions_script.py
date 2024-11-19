"""
Script to organise the raw and processed data from the NZGD into a directory
structure based on geographical regions.
"""

from pathlib import Path

from download_nzgd_data.lib import organise

region_df = organise.find_regions(
    nzgd_index_path=Path(
        "/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_08112024_1017.csv"
    ),
    district_shapefile_path=Path(
        "/home/arr65/data/nzgd/shapefiles/lds-nz-land-districts-SHP_WGS84_EPSG_4326/nz-land-districts.shp"
    ),
    suburbs_shapefile_path=Path(
        "/home/arr65/data/nzgd/shapefiles/lds-nz-suburbs-and-localities-SHP_WGS84_EPSG_4326/nz-suburbs-and-localities.shp"
    ),
    region_classification_output_dir=Path(
        "/home/arr65/data/nzgd/region_classification"
    ),
    num_procs = 8
)


### Organise the raw data from the NZGD into regions
# organise.organise_records_into_regions(
#     processed_data=False,
#     dry_run=False,
#     unorganised_root_dir_to_copy_from=Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd"),
#     organised_root_dir_to_copy_to=Path("/home/arr65/data/nzgd/hypocentre_mirror/nzgd/raw_from_nzgd"),
#     region_df=region_df)

### Organise the processed analysis-ready data from the NZGD into regions
# organise.organise_records_into_regions(
#     processed_data=True,
#     dry_run=False,
#     unorganised_root_dir_to_copy_from=Path(
#         "/home/arr65/data/nzgd/processed_data/cpt/data"
#     ),
#     organised_root_dir_to_copy_to=Path(
#         "/home/arr65/data/nzgd/hypocentre_mirror/nzgd/processed"
#     ),
#     region_df=region_df,
# )


