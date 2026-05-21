import pathlib
import requests
import fsspec
import pandas as pd
import geopandas as gpd
import xarray as xr
import antimeridian
import shapely
from pystac_client import Client


catalog_url = 'https://catalog.maap.eo.esa.int/catalogue/'
catalog = Client.open(catalog_url)
EC_COLLECTION = ['EarthCAREL2Validated_MAAP']

CREDENTIALS_FILE = (pathlib.Path.home() / "credentials.txt" ).resolve()   # Insert the .txt path
io_params = {
    "fsspec_params": {
        "cache_type": "blockcache",
        "block_size": 8 * 1024 * 1024
    },
    "h5py_params": {
        "driver_kwds": {
            "rdcc_nbytes": 8 * 1024 * 1024
        }
    }
}

def load_credentials(file_path=CREDENTIALS_FILE):
    """Read key-value pairs from a credentials file into a dictionary."""
    creds = {}
    if not file_path.exists():
        raise FileNotFoundError(f"Credentials file not found: {file_path}")
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            creds[key.strip()] = value.strip()
    return creds


# --- ESA MAAP API ---

def get_token():
    """Use OFFLINE_TOKEN to fetch a short-lived access token."""
    creds = load_credentials()

    OFFLINE_TOKEN = creds.get("OFFLINE_TOKEN")
    CLIENT_ID = creds.get("CLIENT_ID")
    CLIENT_SECRET = creds.get("CLIENT_SECRET")
    # print(CLIENT_SECRET)

    if not all([OFFLINE_TOKEN, CLIENT_ID, CLIENT_SECRET]):
        raise ValueError("Missing OFFLINE_TOKEN, CLIENT_ID, or CLIENT_SECRET in credentials file")

    url = "https://iam.maap.eo.esa.int/realms/esa-maap/protocol/openid-connect/token"
    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": OFFLINE_TOKEN,
        "scope": "offline_access openid"
    }

    response = requests.post(url, data=data)
    response.raise_for_status()

    response_json = response.json()
    access_token = response_json.get('access_token')

    if not access_token:
        raise RuntimeError("Failed to retrieve access token from IAM response")

    return access_token

token = get_token()

fs = fsspec.filesystem(
    "https", 
    headers={"Authorization": f"Bearer {token}"}, 
    **io_params["fsspec_params"], 
)

def maap_search_to_gdf(search):
    df = pd.DataFrame(
        data={"stac":list(search.items())}
    )
    
    df["granule"] = [f.id[-6:] for f in df.stac]
    df["product"] = [f.id[9:19] for f in df.stac]
    df["baseline"] = [f.id[6:8] for f in df.stac]
    df["date"] = [
        pd.to_datetime(f.id[20:35], format="%Y%m%dT%H%M%S") for f in df.stac
    ]
    df["enclosure_h5"] = [f.assets.get('enclosure_h5').href for f in df.stac]
    df = df.sort_values(["date", "product"]).reset_index(drop=True)

    gdf = gpd.GeoDataFrame(
        df.drop("stac", axis=1), 
        geometry=[
            antimeridian.fix_line_string(
                shapely.LineString(row["stac"].geometry["coordinates"]), 
                great_circle=True,
            )
            for idx, row in df.iterrows()
        ], 
        crs="EPSG:4326"
    )

    return gdf


def search_ec_filename(product, orbit, frame):
    search = catalog.search(
        collections=EC_COLLECTION, 
        filter=f"(productType = '{product}') and orbitNumber = {orbit} and frame = '{frame}'", # For example filter by product type and orbitNumber. Use boolean logic for multi-filter queries
        method = 'GET', # This is necessary 
        max_items=1  # Adjust as needed, given the large amount of products it is recommended to set a limit if especially if you display results in pandas dataframe or similiar
    )
    items = list(search.items())
    if len(items):
        return items[0].assets.get('enclosure_h5').href

    raise ValueError(
        f'No EarthCARE files found for search {product=}, {orbit=}, {frame=}'
    )


from contextlib import contextmanager
@contextmanager
def read_ec_file(filename):
    try:
        f = fs.open(filename)
        dt = xr.open_datatree(
            f, 
            engine="h5netcdf", 
            **io_params["h5py_params"], 
        )
        ds = dt.ScienceData.to_dataset().assign_attrs(
            {
                k:v.item() for k, v in dt.HeaderData.FixedProductHeader.data_vars.items()
            }
        )
        yield ds
    finally:
        f.close()
        try:
            dt.close()
            ds.close()
        except UnboundLocalError:
            pass