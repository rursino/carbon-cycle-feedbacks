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
    global test_ds, basic_test_result, lat, lon, land_vals, ocean_vals

    ds = xr.open_dataset("./../../../data/inversions/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc")

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    land_vals = ds.Terrestrial_flux.values
    ocean_vals = ds.Ocean_flux.values
    land_vals[np.where(land_vals != 0)] = 1 * 365/30
    ocean_vals[np.where(ocean_vals != 0)] = 1 * 365/30

    overlaps = list(zip(*np.where(land_vals + ocean_vals == 2)))
    for index in overlaps:
        ocean_vals[index] = 0

    testData = xr.Dataset(
                {
                'Terrestrial_flux': (('time', 'latitude', 'longitude'),
                                    land_vals),
                'Ocean_flux': (('time', 'latitude', 'longitude'),
                                    ocean_vals)
                },
                coords = {'longitude': lon, 'latitude': lat, 'time': time}
                )

    test_ds = invf.SpatialAgg(testData)
    basic_test_result = test_ds.latitudinal_splits()


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

    basic_land_values = basic_test_result.Earth_Land.values.sum()
    basic_ocean_values = basic_test_result.Earth_Ocean.values.sum()

    assert abs(land_components - basic_land_values) < 1
    assert abs(ocean_components - basic_ocean_values) < 1

def test_earth_area_grid_equals_surface_area():
    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)
    expected_result = test_ds.earth_area_grid(lat,lon).sum()

    assert abs(earth_surface_area - expected_result) < 1

def test_spatial_sum():
    one_month_result = basic_test_result.sel(time = "1993-01")

    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)

    total_flux = (one_month_result.Earth_Land.values +
                  one_month_result.Earth_Ocean.values).sum()

    expected_result = test_ds.earth_area_grid(lat,lon).sum() * 1e-15

    assert abs(total_flux - expected_result) < 1
