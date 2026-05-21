#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import pathlib
import tempfile
import zipfile
from datetime import datetime, timedelta
import numpy as np
from numpy import ma
import xarray as xr

from scipy import ndimage as ndi
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

from sklearn.neighbors import BallTree

import argparse

parser = argparse.ArgumentParser(
    description="Find earthcare overpasses of tracked DCCs"
)
parser.add_argument("file", help="DCC mask file to process", type=str)
args = parser.parse_args()
dcc_file = pathlib.Path(args.file)
assert dcc_file.exists()
dcc_ds = xr.open_dataset(dcc_file)

ec_paths = [
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2b/ACM_CAP_2B/"), 
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2b/ACM_RT__2B/"), 
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2b/AC__TC__2B/"), 
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2a/ATL_EBD_2A/"), 
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2a/CPR_CD__2A/"), 
    pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2a/CPR_FMR_2A/"), 
]

def load_ec_archive(ec_file):
    zf = zipfile.ZipFile(ec_file)

    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        f = list(pathlib.Path(tempdir).glob("*.h5"))[0]
        ec_ds = xr.open_datatree(f).ScienceData.to_dataset().load()
    
    return ec_ds

stats_path = pathlib.Path("/work/scratch-nopw2/wkjones/")
stats_file = list(stats_path.glob("*_S20240810*"))[0]
stats_ds = xr.open_dataset(list(stats_path.glob("*_S20240810*"))[0])

wh_finite_latlon = np.logical_and(
    np.isfinite(dcc_ds.latitude.values.ravel()), 
    np.isfinite(dcc_ds.longitude.values.ravel()), 
)

sev_ll_tree = BallTree(
    np.radians(np.stack(
        [
            dcc_ds.latitude.values.ravel()[wh_finite_latlon], 
            dcc_ds.longitude.values.ravel()[wh_finite_latlon]
        ], axis=1
    )), 
    metric="haversine", 
)

def get_overpass_attrs(
    overpass_slice: xr.Dataset, stats_ds: xr.Dataset, anvil_id: int
) -> dict:
    if anvil_id not in stats_ds.anvil:
        raise ValueError(f'Anvil {anvil_id} not found in stats dataset')

    anvil_ds = stats_ds.sel(anvil=anvil_id)
    anvil_ds = anvil_ds.isel(
        core=anvil_ds.core_anvil_index == anvil_id, 
        thick_anvil_step=anvil_ds.thick_anvil_step_anvil_index == anvil_id
    )

    anvil_is_valid = anvil_ds.thick_anvil_is_valid.item()
    mean_overpass_time = overpass_slice.time.mean().values
    time_from_init = mean_overpass_time - anvil_ds.thick_anvil_start_t.values
    prop_from_init = time_from_init / anvil_ds.thick_anvil_lifetime.values
    anvil_core_count = anvil_ds.anvil_core_count.item()
    anvil_core_intensity = anvil_ds.core_max_cooling_rate.max().item()
    anvil_max_area = anvil_ds.thick_anvil_max_area.item()
    anvil_min_bt = anvil_ds.thick_anvil_bt_min.item()
    anvil_lifetime_stage = ["growing", "maturing", "dissipating"][
        0 if mean_overpass_time < anvil_ds.thick_anvil_min_bt_t else 1 if mean_overpass_time < anvil_ds.thick_anvil_max_area_t else 2
    ]

    nearest_overpass_step = np.abs(anvil_ds.thick_anvil_step_t - mean_overpass_time).idxmin().item()

    from pyproj import Geod

    g = Geod(ellps='GRS80')

    min_overpass_distance = g.inv(
        np.repeat(anvil_ds.thick_anvil_step_lon.sel(thick_anvil_step=nearest_overpass_step).item(), overpass_slice.along_track.size), 
        np.repeat(anvil_ds.thick_anvil_step_lat.sel(thick_anvil_step=nearest_overpass_step).item(), overpass_slice.along_track.size), 
        overpass_slice.longitude.values, 
        overpass_slice.latitude.values
    )[-1].min()/1e3

    return dict(
        anvil_id=anvil_id, 
        anvil_is_valid=str(anvil_is_valid), 
        mean_overpass_time=str(mean_overpass_time),
        time_from_init=int(time_from_init)/1e9, 
        prop_from_init=prop_from_init, 
        anvil_core_count=anvil_core_count, 
        anvil_core_intensity=anvil_core_intensity, 
        anvil_max_area=anvil_max_area, 
        anvil_min_bt=anvil_min_bt, 
        anvil_lifetime_stage=anvil_lifetime_stage, 
        min_overpass_distance=min_overpass_distance, 
    )


def output_colocated_ec_slices(ec_ds_coloc, colocated_anvil_labels, stats_ds, save_path, product, earthcare_file, stats_file, mask_file):
    save_path.mkdir(exist_ok=True, parents=True)
    for anvil_id in np.unique(colocated_anvil_labels):
        if anvil_id > 0:
            overpass_mask = ndi.label(ndi.binary_dilation(
                colocated_anvil_labels==anvil_id, iterations=200
            ))[0]
    
            for label in np.unique(overpass_mask):
                if label > 0:
                    overpass_slice = ec_ds_coloc.isel(along_track = overpass_mask==label)
                    try:
                        overpass_slice = overpass_slice.assign_attrs(get_overpass_attrs(overpass_slice, stats_ds, anvil_id))
                        overpass_slice = overpass_slice.assign_attrs({
                            "earthcare_file":earthcare_file.stem, 
                            "stats_file":stats_file.name,
                            "mask_file":mask_file.name,
                        })
                    except ValueError:
                        pass
                    else:
                        mean_time = datetime.fromisoformat(overpass_slice.mean_overpass_time).strftime("%Y%m%d%H%M%S")
                        save_name = f'{product}_o{earthcare_file.stem.split("_")[-1]}_t{mean_time}_a{anvil_id}.nc'
                        print(save_name)
                        # del overpass_slice.time.attrs["units"]
                        overpass_slice.to_netcdf(save_path / save_name, mode="a")


save_path = pathlib.Path("/work/scratch-nopw2/wkjones/ec_overpasses")
save_path.mkdir(exist_ok=True)

save_path = pathlib.Path("/work/scratch-nopw2/wkjones/ec_overpasses")
save_path.mkdir(exist_ok=True)

ec_path = pathlib.Path("/gws/nopw/j04/eo_shared_data_vol1/satellite/earthcare/L2b/ACM_CAP_2B")


yyyymmdd = dcc_file.name[19:27]
date = datetime.strptime(yyyymmdd, "%Y%m%d")
        
thick_anvil_labels = dcc_ds.thick_anvil_label.load()
thick_anvil_labels = dcc_ds.thick_anvil_label.assign_coords(
    dict(
        x=dcc_ds.thick_anvil_label.x,
        y=dcc_ds.thick_anvil_label.y,
    )
)

for ec_path in ec_paths:
    ec_files = sorted(list((ec_path/date.strftime("%Y/%m/%d")).glob("*.ZIP")))
    for f in ec_files:
        print(f)
        ec_ds = load_ec_archive(f)

        # Check if spatially > 1D

        if len(ec_ds.latitude.shape) > 1:
            if "latitude_active" in ec_ds.data_vars:
                ec_ds = ec_ds.isel(across_track=150)
            else:
                raise ValueError("Cannot handle 2D spitial data yet")
            # latitude = ec_ds.latitude.mean([dim for dim in ec_ds.latitude.dims if dim != "along_track"])
            # longitude = ec_ds.longitude.mean([dim for dim in ec_ds.longitude.dims if dim != "along_track"])
        latitude = ec_ds.latitude
        longitude = ec_ds.longitude
        
        distances, neighbours = sev_ll_tree.query(
            np.radians(np.stack(
                [
                    latitude.values.ravel(), 
                    longitude.values.ravel()
                ], axis=1
            ))
        )
        
        distances = np.degrees(distances.ravel()) * 1.11e2 # Convert distances to km
        neighbours = neighbours.ravel()

        wh_colocated = distances < 10
        ec_ds = ec_ds.isel(along_track=wh_colocated)
        idy_coloc, idx_coloc = np.unravel_index(
            np.where(wh_finite_latlon)[0][neighbours[wh_colocated]], 
            shape=dcc_ds.lat.shape
        )

        y_coloc = dcc_ds.y.values.ravel()[idy_coloc]
        x_coloc = dcc_ds.x.values.ravel()[idx_coloc]

        colocated_anvil_labels = thick_anvil_labels.sel(
            t=ec_ds.time,
            y=xr.DataArray(y_coloc, dims="along_track", coords=dict(along_track=ec_ds.along_track)), 
            x=xr.DataArray(x_coloc, dims="along_track", coords=dict(along_track=ec_ds.along_track)), 
            method="nearest"
        )
        
        if any(colocated_anvil_labels):
            output_colocated_ec_slices(ec_ds, colocated_anvil_labels, stats_ds, save_path/date.strftime("%Y/%m/%d"), ec_path.parts[-1], f, stats_file, dcc_file)

        ec_ds.close()
        del ec_ds


