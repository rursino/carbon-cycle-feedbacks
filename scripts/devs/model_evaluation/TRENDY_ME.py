""" Development and testing of TRENDY ModelEvaluation functions.
"""

""" IMPORTS """
import os
import sys
sys.path.append('./../../core/')
import TRENDY_flux as TRENDYf
import GCP_flux as GCPf

from importlib import reload
reload(TRENDYf);

import numpy as np
import xarray as xr
import pandas as pd
from scipy import stats


""" INPUTS """
MAIN_DIR = './../../../'
fname = (MAIN_DIR +
         "output/TRENDY/spatial/output_all/LPJ-GUESS_S3_nbp/year.nc")


""" EXECUTION """
ds = xr.open_dataset(fname)
df = TRENDYf.ModelEvaluation(ds)



df.regress_timeseries_to_GCP(plot="both")

df.regress_cascading_window_trend_to_GCP(25, plot="both", indep="time")
df.regress_cascading_window_trend_to_GCP(10, plot="both", indep="CO2")


df.compare_trend_to_GCP(True)

df.autocorrelation_plot('South_Land')
