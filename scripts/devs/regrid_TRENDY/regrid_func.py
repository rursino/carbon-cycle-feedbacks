""" Devs and testing of regridding of TRENDY input datasets.
"""


""" IMPORTS """
import xarray as xr
import numpy as np

import sys
sys.path.append('./../../core/')
import TRENDY_flux as TRENDYf


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

interpdf = df_to_interp.nbp.interp(coords={'longitude': df_to_use.longitude,
                            'latitude': df_to_use.latitude})
interpdf.sum()
