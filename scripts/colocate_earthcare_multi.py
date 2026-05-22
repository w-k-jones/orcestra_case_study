#!/home/users/wkjones/miniforge3/envs/tobac_flow/bin/python
import pathlib
import multiprocessing

from datetime import datetime 

import numpy as np
import pandas as pd
import xarray as xr

from colocate_earthcare import main

with open("granules.lst") as f:
    granules = [l[:-1].split(" ") for l in f.readlines()]

if __name__=="__main__":
    # for i, granule in enumerate(granules):
    #     main(i, granule)
    with multiprocessing.get_context('spawn').Pool(24) as p:
        p.starmap(main, enumerate(granules)),
        p.close()
        p.join()