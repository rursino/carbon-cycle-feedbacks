""" Some TRENDY input datasets have cfdatetime as time.
This program contains the developments to convert them into np.datetime64 before
executing spatial output.
"""


""" IMPORTS """
import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd

import sys
sys.path.append('./../../core')
import TRENDY_flux

from importlib import reload
reload(TRENDY_flux)

""" INPUTS """
model = "LPJ-GUESS"
sim = "S1"
fname = f"./../../../data/TRENDY/models/{model}/{sim}/{model}_{sim}_nbp.nc"


""" EXECUTION"""
df = xr.open_dataset(fname)
ds = TRENDY_flux.SpatialAgg(fname)

ds.latitudinal_splits()
