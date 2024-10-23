import organise_data_into_regions_funcs
from pathlib import Path

from load_nzgd_data_script import output_dir

unorganised_downloads_root = Path("/home/arr65/data/nzgd/downloads_and_metadata/unorganised_raw_from_nzgd")
organised_dir = Path("/home/arr65/data/nzgd/downloads_and_metadata/raw_from_nzgd")

nzgd_index_path = Path("/home/arr65/data/nzgd/nzgd_index_files/csv_files/NZGD_Investigation_Report_23102024_1042.csv")
district_shapefile_path = Path("/home/arr65/data/nzgd/shapefiles/lds-nz-land-districts-SHP_WGS84_EPSG_4326/nz-land-districts.shp")
suburbs_shapefile_path = Path("/home/arr65/data/nzgd/shapefiles/lds-nz-suburbs-and-localities-SHP_WGS84_EPSG_4326/nz-suburbs-and-localities.shp")
output_dir = Path("/home/arr65/data/nzgd/region_classification")





region_df = organise_data_into_regions_funcs.find_regions(nzgd_index_path, district_shapefile_path, suburbs_shapefile_path, output_dir)

region_df.fillna("unclassified", inplace=True)

organise_data_into_regions_funcs.organise_records_into_regions(organised_dir, unorganised_downloads_root, region_df)





print()




