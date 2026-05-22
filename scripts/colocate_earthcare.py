#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import pathlib

from datetime import datetime 

import numpy as np
import pandas as pd
import xarray as xr

from products_list import PRODUCTS
from data_access import catalog, EC_COLLECTION, maap_search_to_gdf
from data_preprocessing import get_convex_fov, create_merged_datatree, colocate_tracks
from overpass_processing import output_colocated_ec_slices

save_path = pathlib.Path("/work/scratch-nopw2/wkjones/ec_track_overpasses")
save_path.mkdir(exist_ok=True, parents=True)
slice_save_path = pathlib.Path("/work/scratch-nopw2/wkjones/ec_anvil_overpasses/")
slice_save_path.mkdir(exist_ok=True)

tracking_path = pathlib.Path("/gws/ssde/j25a/esaclim/will/orcestra_linked")
stats_path = pathlib.Path("/gws/ssde/j25a/esaclim/will")
tracking_files = sorted(list(tracking_path.rglob("detected_dccs_*.nc")))
tracks_ds = xr.open_dataset(tracking_files[0])
tracking_bounds = get_convex_fov(tracking_files[0])

def main(i, granules):
    try:
        merged_dt = create_merged_datatree(granules, PRODUCTS, tracking_bounds)
        merged_dt = colocate_tracks(merged_dt, tracking_path)
    except ValueError:
        return None

    save_name = f'EC_merged_overpass_{i:05d}_S{pd.Timestamp(merged_dt.time.min().values).strftime("%Y%m%d_%H%M%S")}_E{pd.Timestamp(merged_dt.time.max().values).strftime("%Y%m%d_%H%M%S")}.nc'

    print(save_name)

    comp = dict(zlib=True, complevel=5, shuffle=True)
    for var in merged_dt.data_vars:
        merged_dt[var].encoding.update(comp)
    for dt in merged_dt.descendants:
        for var in dt.data_vars:
            dt[var].encoding.update(comp)

    merged_dt = merged_dt.load()

    merged_dt.to_netcdf(save_path/save_name)

    # Now process individual anvil overpasses:
    stats_file = list(stats_path.glob("*_S20240810*"))[0]
    stats_ds = xr.open_dataset(stats_file)
    
    for overpass_slice in output_colocated_ec_slices(
        merged_dt, merged_dt.thin_anvil_label, stats_ds, stats_file
    ):
        mean_time = pd.Timestamp(overpass_slice.mean_overpass_time.item()).strftime("%Y%m%d_%H%M%S")
        save_name = f'EC_anvil_overpass_T{mean_time}_A{overpass_slice.anvil_id.item():06d}.nc'
        
        print(save_name)
        overpass_slice.to_netcdf(slice_save_path/save_name)

if __name__=="__main__":
    print("Parsing arguments", flush=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    parser.add_argument('granules', nargs='*')
    args = parser.parse_args()
    
    print("Running main", flush=True)
    main(args.i, args.granules)