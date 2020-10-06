""" Development of cascading window trend functions.
"""

""" IMPORTS """
import sys
from core import inv_flux as invf
from core import trendy_flux as TRENDYf

from importlib import reload
reload(invf);
reload(TRENDYf);

import xarray as xr
import pandas as pd
from datetime import datetime
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

""" INPUTS """
fname = "./../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
df = xr.open_dataset(fname)
co2 = pd.read_csv('./../../../data/co2/co2_year.csv', index_col='Year').CO2

trendy_df = xr.open_dataset("./../../../output/TRENDY/spatial/output_all/VISIT_S3_nbp/year.nc")


""" DEVS """
invf.Analysis(df).cascading_window_trend()
TRENDYf.Analysis(trendy_df).cascading_window_trend().plot()
