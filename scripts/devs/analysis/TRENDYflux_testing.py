""" Test and development of the TRENDYf.Analysis functions.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import sys
sys.path.append("./../../core")
import TRENDY_flux as TRENDYf

from importlib import reload
reload(TRENDYf)

fname = f"./../../../output/TRENDY/spatial/mean_all/S3/year.nc"

ds = xr.open_dataset(fname)

df = TRENDYf.Analysis(ds)

df.plot_timeseries("South_Land", slice("1770", "1792"))

df.cascading_window_trend(indep="CO2", plot=True, window_size=20, include_pearson=False,
                            include_linreg=True);

df.psd("South_Land", fs=12, plot=True)


x = df.deseasonalise("Earth_Land")

plt.plot(df.data.Earth_Land.values)
plt.plot(x)

plt.plot(df.data.time, df.data.Earth_Land.values)
plt.plot(df.data.time, df.bandpass("Earth_Land", fc=1/(20)))

plt.plot(df.data.time, df.data.Earth_Land.values)
plt.plot(df.data.time, df.bandpass("Earth_Land", fc=1/(20*12), deseasonalise_first=True))
