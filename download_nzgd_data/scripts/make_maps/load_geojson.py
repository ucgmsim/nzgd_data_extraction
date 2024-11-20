import geopandas as gpd
from pathlib import Path

load_dir = Path("/home/arr65/data/nzgd/resources")

vs30_from_data_df = gpd.read_file(load_dir / "vs30_from_data.geojson")

print()

# # Load the GeoTIFF file
# with rasterio.open(load_dir / "vs30map_data_2023_geotiff/vs30map_data/combined.tif") as src:
#     bounds = src.bounds
#     image = reshape_as_image(src.read())
#
# # Add the GeoTIFF as an overlay
# folium.raster_layers.ImageOverlay(
#     image=image,
#     bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
#     opacity=0.6
# ).add_to(fmap)