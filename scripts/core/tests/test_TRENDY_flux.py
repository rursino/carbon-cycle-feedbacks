""" pytest: inv_flux module.
"""


""" IMPORTS """
import sys
sys.path.append("./../")
import TRENDY_flux as TRENDYf

import numpy as np
import xarray as xr

import pytest

""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global original_ds, test_ds, output_ds
    global basic_test_result, lat, lon, land_vals, ocean_vals

    fname = "./../../../data/TRENDY/models/LPJ-GUESS/S0/LPJ-GUESS_S0_cVeg.nc"

    ds = xr.open_dataset(fname)

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    vals = ds.cVeg.values
    vals[np.isnan(vals)] = 0
    vals[vals > 0] = 1

    testData = xr.Dataset(
                {
                'cVeg': (('time', 'latitude', 'longitude'),
                                    vals)
                },
                coords = {'longitude': lon, 'latitude': lat, 'time': time}
                )

    original_ds = TRENDYf.SpatialAgg(ds)
    test_ds = TRENDYf.SpatialAgg(testData)

    basic_test_result = test_ds.latitudinal_splits()

    #Output dataframe
    output_fname = './../../../output/TRENDY/spatial/output_all/LPJ-GUESS_S1_nbp/year.nc'
    output_ds = xr.open_dataset(output_fname)

""" TESTS """
def test_check_instance():
    assert isinstance(basic_test_result, xr.Dataset)

def test_regions_add_to_global():

    def differences(dataset):

        components = (dataset.South.values +
                           dataset.Tropical.values +
                           dataset.North.values).sum()

        values = dataset.Earth.values.sum()
        return np.array([abs(components - values)])

    assert np.all(differences(basic_test_result) < 1)
    assert np.all(differences(test_ds.latitudinal_splits(23)) < 1)
    assert np.all(differences(original_ds.latitudinal_splits()) < 1)
    assert np.all(differences(original_ds.latitudinal_splits(23)) < 1)
    assert np.all(differences(output_ds) < 1)

def test_earth_area_grid_equals_surface_area():
    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)
    expected_result = test_ds.earth_area_grid(lat,lon).sum()

    assert abs(earth_surface_area - expected_result) < 1

def test_spatial_sum():

    def differences(dataset):
        one_month_result = dataset.sel(time = "1993-01")
        earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)

        total_flux = one_month_result.Earth.values.sum()
        expected_result = test_ds.earth_area_grid(lat,lon).sum() * 1e-15

        return abs(total_flux - expected_result)

    assert differences(basic_test_result) < 1
    assert differences(test_ds.latitudinal_splits(23)) < 1
