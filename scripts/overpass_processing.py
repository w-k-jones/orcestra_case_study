import numpy as np
import xarray as xr

def select_anvil(ds, anvil):
    ds = ds.sel(anvil=anvil)
    ds = ds.isel(thick_anvil_step=ds.thick_anvil_step_anvil_index==anvil)
    ds = ds.isel(thin_anvil_step=ds.thin_anvil_step_anvil_index==anvil)
    ds = ds.isel(core=ds.core_anvil_index==anvil)
    ds = ds.isel(core_step=np.isin(ds.core_step_core_index, ds.core))
    return ds

def get_overpass_attrs(
    overpass_slice: xr.Dataset, stats_ds: xr.Dataset, anvil_id: int
) -> dict:
    if anvil_id not in stats_ds.anvil:
        raise ValueError(f'Anvil {anvil_id} not found in stats dataset')

    anvil_ds = select_anvil(stats_ds, anvil_id)

    thick_anvil_is_valid = anvil_ds.thick_anvil_is_valid.item()
    thin_anvil_is_valid = anvil_ds.thin_anvil_is_valid.item()
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
        thick_anvil_is_valid=thick_anvil_is_valid, 
        thin_anvil_is_valid=thin_anvil_is_valid, 
        mean_overpass_time=mean_overpass_time,
        time_from_init=int(time_from_init)/1e9, 
        prop_from_init=prop_from_init, 
        anvil_core_count=anvil_core_count, 
        anvil_core_intensity=anvil_core_intensity, 
        anvil_max_area=anvil_max_area, 
        anvil_min_bt=anvil_min_bt, 
        anvil_lifetime_stage=anvil_lifetime_stage, 
        min_overpass_distance=min_overpass_distance, 
    )

def output_colocated_ec_slices(ec_ds_coloc, colocated_anvil_labels, stats_ds, stats_file):
    for anvil_id in np.unique(colocated_anvil_labels):
        if anvil_id > 0:
            wh_overpass = np.where(colocated_anvil_labels==anvil_id)[0]
            overpass_slice = ec_ds_coloc.isel(along_track = slice(
                np.maximum(0, wh_overpass.min()-100),
                np.minimum(colocated_anvil_labels.size, wh_overpass.max()+100),
            ))
            if len(overpass_slice.along_track) > 0:
                try:
                    for k, v in get_overpass_attrs(overpass_slice, stats_ds, anvil_id).items():
                        overpass_slice[k] = v
                    overpass_slice.attrs["stats_file"] = stats_file.name
                    overpass_slice["/dcc_properties"] = select_anvil(stats_ds, overpass_slice.anvil_id)
                except ValueError:
                    pass
                else:
                    yield overpass_slice
                    