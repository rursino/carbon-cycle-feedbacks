""" Development of functionalities to perform linear regression (using scipy)
with datetime objects (such as np.datetime64 and pd.DatetimeIndex).
"""

""" IMPORTS """
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from datetime import datetime

import sys
sys.path.append("./../../core/")
import inv_flux as invf


""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/Rayner/year.nc"


""" FUNCTIONS """
def to_numerical(data):
    return (pd.to_numeric(data) /
            (3600 * 24 * 365 *1e9) + 1969
            )


""" EXECUTION """
ds = xr.open_dataset(fname)

# with np.datetime64[ns]
x = to_numerical(ds.time.values)
y = ds.Earth_Land.values

stats.linregress(x,y)
x = sm.add_constant(x)
sm.OLS(y,x).fit().summary()

# with pd.DatetimeIndex
xx = pd.to_datetime(ds.time.values)
xx = to_numerical(xx)
np.all(xx == x)
