
import pathlib

import numpy as np
import pandas as pd
import xarray as xr

from products_list import PRODUCTS
from data_access import catalog, EC_COLLECTION, maap_search_to_gdf
from data_preprocessing import get_convex_fov, create_merged_datatree, colocate_tracks

save_path = pathlib.Path("/work/scratch-nopw2/wkjones/ec_track_overpasses")
save_path.mkdir(exist_ok=True, parents=True)

tracking_path = pathlib.Path("/gws/ssde/j25a/esaclim/will/orcestra_linked")
tracking_files = sorted(list(tracking_path.rglob("detected_dccs_*.nc")))
tracks_ds = xr.open_dataset(tracking_files[8])

tracking_bounds = get_convex_fov(tracking_files[0])

# Find all CPR granules
gdf = maap_search_to_gdf(
    catalog.search(
        collections=EC_COLLECTION, 
        datetime=("2024-08-10", "2024-09-30"), 
        bbox=tracking_bounds.bounds, 
        filter="(productType = 'CPR_FMR_2A')",
        method = 'GET', # This is necessary 
    )
)

granule_list = [
    df.granule.tolist() for _, df in gdf.groupby((gdf.date.diff() > np.timedelta64(15, "m")).cumsum())
]

for i, granules in enumerate(granule_list[443:]):
    try:
        merged_dt = create_merged_datatree(granules, PRODUCTS, tracking_bounds)
    except ValueError:
        pass
    else:
        merged_dt = colocate_tracks(merged_dt, tracking_path)
    
        save_name = f'EC_merged_overpass_{i+443:05d}_S{pd.Timestamp(merged_dt.time.min().values).strftime("%Y%m%d_%H%M%S")}_E{pd.Timestamp(merged_dt.time.max().values).strftime("%Y%m%d_%H%M%S")}.nc'

        print(f'{i+443}: {save_name}')

        comp = dict(zlib=True, complevel=5, shuffle=True)
        for var in merged_dt.data_vars:
            merged_dt[var].encoding.update(comp)
        for dt in merged_dt.descendants:
            for var in dt.data_vars:
                dt[var].encoding.update(comp)
    
        merged_dt.to_netcdf(save_path/save_name)
    