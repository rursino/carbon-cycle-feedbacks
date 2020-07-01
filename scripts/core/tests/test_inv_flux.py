""" pytest: inv_flux module.
"""


""" IMPORTS """
import sys
sys.path.append("./../")
import inv_flux as invf

import numpy as np
import xarray as xr
from cftime import Datetime360Day

""" SETUP """
ds, basic_result = None, None

def setup_module(module):
    print('--------------------setup--------------------')
    global ds, basic_result

    ds = xr.open_dataset("./../../../data/inversions/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc")

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values
    vals = np.ones([468, 180, 360])

    testData = xr.Dataset(
                {
                'Terrestrial_flux': (('time', 'latitude', 'longitude'), vals),
                'Ocean_flux': (('time', 'latitude', 'longitude'), vals)
                },
                coords = {'longitude': lon, 'latitude': lat, 'time': time}
                )

    ds = invf.SpatialAgg(testData)
    basic_result = ds.spatial_integration()


""" TESTS """
def test_check_instance():
    assert isinstance(basic_result, xr.Dataset)

def test_spatial_integration():
