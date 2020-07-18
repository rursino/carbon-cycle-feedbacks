""" Development of cascading window trend functions for both time and CO2.
"""

""" IMPORTS """
import sys
sys.path.append("./../../core")
import inv_flux as invf

from importlib import reload
reload(invf)

import xarray as xr
import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd

""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/JENA_s76/month.nc"
ds = xr.open_dataset(fname)
df = invf.Analysis(ds)
df.time_resolution

CO2fname = "./../../../data/CO2/co2_month_weighted.csv"
pd.read_csv(CO2fname, index_col=['Year', 'Month'])['CO2']

""" DEVS """
df.cascading_window_trend(plot=True, window_size=15)

ds
