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
def setup_module(module):
    print('--------------------setup--------------------')
    global test_ds, basic_test_result

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

    test_ds = invf.SpatialAgg(testData)
    basic_test_result = test_ds.spatial_integration()


""" TESTS """
def test_check_instance():
    assert isinstance(basic_test_result, xr.Dataset)

def test_regions_add_to_global():
    land_components = (basic_test_result.South_Land.values +
                       basic_test_result.Tropical_Land.values +
                       basic_test_result.North_Land.values).sum()
    ocean_components = (basic_test_result.South_Ocean.values +
                        basic_test_result.Tropical_Ocean.values +
                        basic_test_result.North_Ocean.values).sum()

    assert land_components == basic_test_result.Earth_Land.values.sum()
    assert ocean_components == basic_test_result.Earth_Ocean.values.sum()

def test_spatial_sum():
    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)

    total_flux = (basic_test_result.Earth_Land.values +
                  basic_test_result.Earth_Ocean.values).sum()

    assert total_flux == earth_surface_area * 1e-15
