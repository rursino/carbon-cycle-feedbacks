""" Development of cascading window trend functions for both time and CO2.
"""

""" IMPORTS """
import sys
from core import inv_flux as invf

from importlib import reload
reload(invf)

import xarray as xr
import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/JENA_s76/year.nc"


""" DEVS """
ds = xr.open_dataset(fname)
df = invf.Analysis(ds)

plt.plot(df.data.South_Land.values)
plt.plot(df.deseasonalise("South_Land"))

df.cascading_window_trend("time", "North_Land", plot=True, window_size=15,
                            include_linreg=True)

df.cascading_window_trend("CO2", "North_Land", plot=True, window_size=15)
