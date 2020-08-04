""" Devs and testing of regridding of TRENDY input datasets.
"""


""" IMPORTS """
import xarray as xr
import numpy as np

import sys
sys.path.append('./../../core/')
import TRENDY_flux as TRENDYf

from importlib import reload
reload(TRENDYf)


""" INPUTS """
CLASSfname = './../../../data/TRENDY/models/CLASS-CTEM/S1/CLASS-CTEM_S1_nbp.nc'
JSBACHfname = './../../../data/TRENDY/models/JSBACH/S1/JSBACH_S1_nbp.nc'
OCNfname = './../../../data/TRENDY/models/OCN/S1/OCN_S1_nbp.nc'
LPJfname = './../../../data/TRENDY/models/LPJ-GUESS/S1/LPJ-GUESS_S1_nbp.nc'
CABLEfname = './../../../data/TRENDY/models/CABLE-POP/S1/CABLE-POP_S1_nbp.nc'


""" EXECUTION """
df_CLASS = xr.open_dataset(CLASSfname)
df_JSBACH = xr.open_dataset(JSBACHfname)
df_OCN = xr.open_dataset(OCNfname)

df_to_interp = df_CLASS
df_to_use = df_OCN

# Check that np arange arrays equal to OCN lat-lon arrays.
np.all(np.arange(-89.5, 90.5, 1) == df_to_use.latitude.values)
np.all(np.arange(-179.5, 180.5, 1) == df_to_use.longitude.values)

# Use the xr.DataArray.interp function as shown below.
interpdf = df_to_interp.nbp.interp(coords={'longitude': df_to_use.longitude,
                            'latitude': df_to_use.latitude})

# Sum of fluxes will be slightly different as a result of interpolation.
interpdf.sum()


""" TESTING """
# Test this fuctionality as implemented when a TRENDYf.SpatialAgg class is
# initialised.
Tdf = TRENDYf.SpatialAgg(CLASSfname)
Tdf.regional_cut(lats=(-30, 30), lons=(0,40))
