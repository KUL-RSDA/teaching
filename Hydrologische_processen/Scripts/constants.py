# --- modules ---
import os, sys
import geopandas as gpd
import rootutils

# --- hard coded paths ---
ROOT_PATH = rootutils.find_root(search_from=__file__, indicator=".git")
sys.path.append(str(ROOT_PATH))
ROOTDIR_DATA = os.path.join(ROOT_PATH, "Data")
sys.path.append(ROOTDIR_DATA)

# %% Constants
MILLIMETER_PER_M = 1000
SECONDS_PER_DAY = 24 * 3600
SQUARE_METERS_PER_SQUARE_KILOMETER = 1e6
INITIAL_STORAGE_DEMER = 550000000

gdf_Demer = gpd.read_file(os.path.join(ROOTDIR_DATA, "QGIS", "Demer_stroomgebied.shp"))
CATCHMENT_AREA_DEMER = gdf_Demer.geometry.area.iloc[0] / SQUARE_METERS_PER_SQUARE_KILOMETER  # m2 to km^2

gdf_Zwalm = gpd.read_file(os.path.join(ROOTDIR_DATA, "processed/data_Zwalm", "Zwalm_catchment.shp"))
CATCHMENT_AREA_ZWALM = gdf_Zwalm.geometry.area.iloc[0] / SQUARE_METERS_PER_SQUARE_KILOMETER  # m2 to km^2