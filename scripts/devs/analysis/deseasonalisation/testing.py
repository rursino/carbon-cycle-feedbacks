""" Test new deasonalisation function in TRENDYF and INVF.
"""

""" IMPORTS """
import sys
sys.path.append('./../../../core/')
import TRENDY_flux as TRENDYf
import inv_flux as invf

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


""" INPUTS """
INV_INPUT = './../../../../output/inversions/spatial/output_all/JENA_s76/month.nc'
TRENDY_INPUT = './../../../../output/TRENDY/spatial/output_all/CABLE-POP_S1_nbp/month.nc'


""" EXECUTION """
# inversions
invdf = xr.open_dataset(INV_INPUT)
df = invf.Analysis(invdf)

plt.plot(df.data.Earth_Land.values)
plt.plot(df.deseasonalise('Earth_Land'))

# trendy
trendydf = xr.open_dataset(TRENDY_INPUT)
tdf = invf.Analysis(trendydf)

plt.plot(tdf.data.Earth_Land.values)
plt.plot(tdf.deseasonalise('Earth_Land'))
plt.xlim([50,2900])

# bandpass functions
plt.plot(tdf.bandpass('Earth_Land', fc=1/25, fs=12, btype='low'))


""" Test: successful."""
