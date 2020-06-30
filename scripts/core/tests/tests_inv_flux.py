"""
"""


""" IMPORTS """
import sys
sys.path.append("./../")
import inv_flux as invf

import numpy as np
import xarray as xr
from cftime import Datetime360Day


""" INPUTS """
t = xr.open_dataset("./../../../data/inversions/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc")
t.Terrestrial_flux.isel(time=[0,1],
                        lat
                        )


lon = np.array([0], dtype=float)
lat = np.arange(-89.5, 89.5, 60, dtype=float)
time = np.array([Datetime360Day(1979,12,16), Datetime360Day(1980,1,16)])

land = np.ones([len(lat), len(lon), len(time)]).squeeze()
land
ocean = land + 1
ocean

testData = xr.Dataset(
    data_vars = {'Terrestrial_flux': (['latitude', 'longitude', 'time'], land),
                 'Ocean_flux': (['latitude', 'longitude', 'time'], ocean)},
    coords = {'latitude': (('latitude'), lat),
              'longitude': (('longitude'), lon),
              'time': (('time'), time)
             }
)

ds = invf.SpatialAgg(testData)

ds.spatial_integration()

def test_spatial_integration():
