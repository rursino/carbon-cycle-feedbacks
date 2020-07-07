""" pytest: inv_flux module.
"""


""" IMPORTS """
import sys
sys.path.append("./../")
import TEMP

import numpy as np
import xarray as xr

import pytest

""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global original_ds, test_ds
    global basic_test_result, lat, lon, land_vals, ocean_vals

    fname = "./../../../data/temp/crudata/HadSST.3.1.1.0.median.nc"
    ds = xr.open_dataset(fname)

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    vals = ds.sst.values
    vals = np.ones(vals.shape)
    vals

    testData = xr.Dataset(
                {
                'sst': (('time', 'latitude', 'longitude'), vals)
                },
                coords = {'longitude': lon, 'latitude': lat, 'time': time}
                )

    original_ds = TEMP.SpatialAve(ds)
    test_ds = TEMP.SpatialAve(testData)

    basic_test_result = test_ds.latitudinal_splits()


""" TESTS """
def test_check_instance():
    assert isinstance(basic_test_result, xr.Dataset)

def test_regions_add_to_global():

    def differences(dataset):

        components = np.array((dataset.South.values,
                               dataset.North.values,
                               dataset.Tropical.values)).mean()
        values = dataset.Earth.values.mean()

        return np.array(abs(components - values))

    assert np.all(differences(basic_test_result) < 1)
    assert np.all(differences(test_ds.latitudinal_splits(23)) < 1)
    assert np.all(differences(original_ds.latitudinal_splits()) < 1)
    assert np.all(differences(original_ds.latitudinal_splits(23)) < 1)

def test_spatial_sum():

    def differences(dataset):
        one_month_result = dataset.sel(time = "1999-12")
        total_flux = one_month_result.Earth.values.sum()

        return abs(total_flux - 1.0)

    assert differences(basic_test_result) == 0
    assert differences(test_ds.latitudinal_splits(23)) == 0
