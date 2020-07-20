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
import matplotlib.pyplot as plt

""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/JENA_s76/year.nc"


""" DEVS """
ds = xr.open_dataset(fname)
df = invf.Analysis(ds)

df.cascading_window_trend("raw", "North_Land", plot=True, window_size=15)

df.cascading_window_trend("time", "Tropical_Land", plot=False, include_pearson=True)[1]
