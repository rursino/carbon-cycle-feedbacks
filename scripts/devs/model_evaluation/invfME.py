""" Testing and development of the model evaluation functionality for the
inversion datasets.
"""

""" IMPORTS """
import os
import sys
from core import inv_flux as invf
from core import GCP_flux as GCPf

from importlib import reload
reload(invf)

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats


""" INPUTS """
MAIN_DIR = './../../../'
fname = (MAIN_DIR +
         "output/inversions/spatial/output_all/CAMS/year.nc")


""" EXECUTION """
ds = xr.open_dataset(fname)
df = invf.ModelEvaluation(ds)


df._time_to_CO2(df.data.time.values)

df.regress_timeseries_to_GCP("land", plot="both")

df.regress_cascading_window_trend_to_GCP(25, "land", plot="both", indep="time")
df.regress_cascading_window_trend_to_GCP(10, "land", plot="both", indep="CO2")
