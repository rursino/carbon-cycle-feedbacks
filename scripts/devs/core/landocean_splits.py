""" Developments to split TEMP output data to land and ocean and learn whether
this can be applied universally throughout the TRENDY and inversion inputs or
they need to applied uniquely for each dataset.
"""

""" IMPORTS """
import numpy as np
import xarray as xr

import sys
sys.path.append("./../../core/")
import TRENDY_flux as TRENDYf
import inv_flux as invf
import TEMP


""" INPUTS """
TEMPDF = TEMP.SpatialAve("./../../../data/temp/crudata/HadCRUT.4.6.0.0.median.nc")
invfDF = invf.SpatialAgg("./../../../data/inversions/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc")
TRENDYfDF = TRENDYf.SpatialAgg("./../../../data/TRENDY/models/CLASS-CTEM/S1/CLASS-CTEM_S1_nbp.nc")


condition = np.isnan(TRENDYfDF.data.isel(time=0).nbp.values)

TRENDYfDF.data.isel(time=0).nbp.values.shape
TEMPDF.data.isel(time=0).temperature_anomaly.shape


""" FUNCTIONS """



""" EXECUTION """
