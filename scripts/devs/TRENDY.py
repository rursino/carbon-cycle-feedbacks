"""Development of the TRENDY_flux.py script.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("./../core")
import TRENDY_flux as TRENDYf

from importlib import reload
reload(TRENDYf);


""" INPUTS """
model = "CLASS-CTEM"
sim = "S1"
fname = f"./../../data/TRENDY/models/{model}/{sim}/{model}_{sim}_nbp.nc"

fname
""" FUNCTIONS """


""" EXECUTION """
df = TRENDYf.SpatialAgg(fname)
df.time_range(slice_obj=True)

res = df.regional_cut((30,60), (100, 120), '1900', '1905')

df.latitudinal_splits(lat_split=40, start_time='1900', end_time='1908')

var = list(df.data.var())[0]
