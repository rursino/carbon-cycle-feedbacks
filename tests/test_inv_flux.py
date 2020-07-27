""" pytest: inv_flux module.
"""


""" IMPORTS """
import os
CURRENT_DIR = ''#os.path.dirname(__file__)

import sys
sys.path.append(CURRENT_DIR + "./../scripts/core")
import inv_flux as invf

import numpy as np
import xarray as xr
from cftime import Datetime360Day

import pytest

""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global original_ds, test_ds, month_output, year_output
    global basic_test_result, lat, lon, land_vals, ocean_vals

    dir = CURRENT_DIR + "./../data/inversions/"
    fname = "fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
    ds = xr.open_dataset(dir + fname)

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

    original_ds = invf.SpatialAgg(ds)
    test_ds = invf.SpatialAgg(testData)

    basic_test_result = test_ds.latitudinal_splits()

    #Output dataframe
    output_dir = CURRENT_DIR + './../output/inversions/spatial/output_all/CAMS/'
    month_output = xr.open_dataset(output_dir + 'month.nc')
    year_output = xr.open_dataset(output_dir + 'year.nc')

""" TESTS """
def test_check_instance():
    assert isinstance(basic_test_result, xr.Dataset)

def test_regions_add_to_global():

    def differences(dataset):

        land_components = (dataset.South_Land.values +
                           dataset.Tropical_Land.values +
                           dataset.North_Land.values).sum()
        ocean_components = (dataset.South_Ocean.values +
                            dataset.Tropical_Ocean.values +
                            dataset.North_Ocean.values).sum()

        land_values = dataset.Earth_Land.values.sum()
        ocean_values = dataset.Earth_Ocean.values.sum()

        return np.array([abs(land_components - land_values),
               abs(ocean_components - ocean_values)])

    assert np.all(differences(basic_test_result) < 1)
    assert np.all(differences(test_ds.latitudinal_splits(23)) < 1)
    assert np.all(differences(original_ds.latitudinal_splits()) < 1)
    assert np.all(differences(original_ds.latitudinal_splits(23)) < 1)
    assert np.all(differences(month_output) < 1)

def test_earth_area_grid_equals_surface_area():
    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)
    expected_result = test_ds.earth_area_grid(lat,lon).sum()

    assert abs(earth_surface_area - expected_result) < 1

def test_spatial_sum():

    def differences(dataset):
        one_month_result = dataset.sel(time = "1993-01")

        earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)

        total_flux = (one_month_result.Earth_Land.values +
                      one_month_result.Earth_Ocean.values).sum()

        expected_result = test_ds.earth_area_grid(lat,lon).sum() * 1e-15

        return abs(total_flux - expected_result)

    assert differences(basic_test_result) < 1
    assert differences(test_ds.latitudinal_splits(23)) < 1

def output_equals_result():

    assert month_output == basic_test_result

def months_add_to_years():

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
