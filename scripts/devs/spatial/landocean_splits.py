""" Tests to check viability of using HadSST and CRUTEM datasets as land and
ocean temperature data for further analysis by comparing it to HadCRUT (which
combines both land and ocean counterpart datasets).
Originally, it was proposed to split HadCRUT data into lat-lon grids which were
land and ocean. However, the HadSST and CRUTEM datasets deem this to be
unncessary.
"""


""" IMPORTS """
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats

import sys
sys.path.append("./../../core/")
import TRENDY_flux as TRENDYf
import inv_flux as invf
import TEMP

from importlib import reload
reload(TEMP)


""" INPUTS """
TEMPdir = "./../../../data/temp/crudata/"
fCRUTEM = TEMPdir + "CRUTEM.4.6.0.0.anomalies.nc"
fHadSST = TEMPdir + "HadSST.3.1.1.0.median.nc"
fHadCRUT = TEMPdir + "HadCRUT.4.6.0.0.median.nc"
TEMPDF = TEMP.SpatialAve(fHadCRUT)

invfDF = invf.SpatialAgg("./../../../data/inversions/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc")
TRENDYfDF = TRENDYf.SpatialAgg("./../../../data/TRENDY/models/CLASS-CTEM/S1/CLASS-CTEM_S1_nbp.nc")

fHadCRUT.split("/")[-1].split(".")[0]


""" DEVS """
dfHadSST = TEMP.SpatialAve(fHadSST)
dfHadCRUT = TEMP.SpatialAve(fHadCRUT)
dfCRUTEM = TEMP.SpatialAve(fCRUTEM)

def array_values(df, time="1959-12"):
    return df.data[df.var].sel(time=time).values.squeeze()

av_HadSST = array_values(dfHadSST)
av_HadCRUT = array_values(dfHadCRUT)
av_CRUTEM = array_values(dfCRUTEM)


def mapplot(lons, lats, data):
    fig = plt.figure()
    ax = fig.add_subplot(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.contourf(lons, lats, data, colorbar=True)

mapplot(dfHadSST.data.longitude, dfHadSST.data.latitude, array_values(dfHadSST))
mapplot(dfHadCRUT.data.longitude, dfHadCRUT.data.latitude, array_values(dfHadCRUT))
mapplot(dfCRUTEM.data.longitude, dfCRUTEM.data.latitude, array_values(dfCRUTEM))

mapplot(dfHadCRUT.data.longitude,
        dfHadCRUT.data.latitude,
        av_HadCRUT - av_HadSST)

mapplot(dfHadCRUT.data.longitude,
        dfHadCRUT.data.latitude,
        av_HadCRUT - av_CRUTEM)

diffLand = (av_HadCRUT - av_CRUTEM)[np.isfinite((av_HadCRUT - av_CRUTEM))]
diffOcean = (av_HadCRUT - av_HadSST)[np.isfinite((av_HadCRUT - av_HadSST))]

plt.hist(diffLand, density=True)
stats.norm.pdf(diffLand).mean(), stats.norm.pdf(diffLand).std()

plt.hist(diffOcean, density=True)
stats.norm.pdf(diffOcean).mean(), stats.norm.pdf(diffOcean).std()
