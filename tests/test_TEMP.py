""" pytest: inv_flux module.
"""


""" IMPORTS """
import os
CURRENT_DIR = os.path.dirname(__file__)

import sys
sys.path.append(CURRENT_DIR + "./../scripts/core")
import TEMP

import numpy as np
import xarray as xr

import pytest

""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global original_ds, test_ds, month_output, year_output
    global basic_test_result, lat, lon, land_vals, ocean_vals

    fname = CURRENT_DIR + "./../data/temp/crudata/HadSST.3.1.1.0.median.nc"
    ds = xr.open_dataset(fname)

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    # Set up xr.Dataset with same shape as ds but only has values 0 if lat-lon
    # position is over ocean and 1 if over land.
    vals = ds.sst.values
    vals = np.ones(vals.shape)
    vals
    testData = xr.Dataset(
                {
                'sst': (('time', 'latitude', 'longitude'), vals)
                },
                coords = {'longitude': lon, 'latitude': lat, 'time': time}
                )

    # Initialise SpatialAgg instances with original and test datasets.
    original_ds = TEMP.SpatialAve(ds)
    test_ds = TEMP.SpatialAve(testData)

    # Basic test result array to avoid calling latitudinal_splits multiple times.
    basic_test_result = test_ds.latitudinal_splits()

    # Output dataframe.
    output_dir = CURRENT_DIR + './../output/TEMP/spatial/output_all/HadSST/'
    month_output = xr.open_dataset(output_dir + 'month.nc')
    year_output = xr.open_dataset(output_dir + 'year.nc')


""" TESTS """
def test_check_instance():
    """ Check that basic_test_result is a xr.Dataset.
    """
    assert isinstance(basic_test_result, xr.Dataset)

def test_regions_add_to_global():
    """ Test that south, tropical and north regional fluxes add to the global
    fluxes.
    """

    def differences(dataset):
        """ Returns the difference between sum of regional components and
        global values.
        """

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
    """ Check that sum of fluxes in test_ds equals the entire area of earth (or
    of all grids created from earth_area_grid func).
    """

    def differences(dataset):
        one_month_result = dataset.sel(time = "1999-12")
        total_flux = one_month_result.Earth.values.sum()

        return abs(total_flux - 1.0)

    assert differences(basic_test_result) == 0
    assert differences(test_ds.latitudinal_splits(23)) == 0

def test_output_equals_result():
    """ Check that output dataset equals basic_test_result array created in setup.
    """

    assert month_output == basic_test_result

def test_months_add_to_years():
    """ Check that all months add up to corresponding year. Check this for many
    years and various variables.
    """

    def sums(year, variable):
        month_time = slice(f"{year}-01", f"{year}-12")
        month_sum = (month_output
                        .sel(time=month_time)[variable].values
                        .sum()
                    )
        year_sum = (year_output
                        .sel(time=year)[variable].values
                        .sum()
                   )

        return month_sum, year_sum

    args = [
                ("1990", "Earth_Land"),
                ("1800", "Earth_Land"),
                ("1945", "South_Land"),
                ("2006", "North_Land"),
                ("1976", "Tropical_Land"),

           ]
    for arg in args:
        assert np.subtract(*sums(*arg)) == 0
