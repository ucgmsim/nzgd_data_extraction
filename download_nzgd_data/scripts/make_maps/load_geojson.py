import geopandas as gpd
from pathlib import Path

load_dir = Path("/home/arr65/data/nzgd/resources")

vs30_from_data_df = gpd.read_file(load_dir / "vs30_from_data.geojson")

print()