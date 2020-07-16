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

""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
ds = xr.open_dataset(fname)
df = invf.Analysis(ds)


""" DEVS """
df.cascading_window_trend("Earth_Land", plot=True, window_size=10)


df.time[0:11]
time_list = [datetime.strptime(time.strftime('%Y-%m'), '%Y-%m') for
             time in df.data.time.values]
pd.to_datetime(time_list).strftime('%Y').__dir__()

df.data.time.toordinal()
x = pd.to_datetime(df.time)#.strftime("%Y")
y = np.arange(len(x))

datetime.toordinal(x)

stats.linregress(x,y)
