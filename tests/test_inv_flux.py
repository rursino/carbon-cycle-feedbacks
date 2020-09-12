""" pytest: inv_flux module.
"""


""" IMPORTS """
from core import inv_flux as invf

import numpy as np
import xarray as xr
from cftime import Datetime360Day
import pandas as pd

import pytest

""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global original_ds, test_ds, month_output, year_output
    global basic_test_result, lat, lon, land_vals, ocean_vals, co2_year, co2_month

    CURRENT_DIR = './'

    co2_year = pd.read_csv('./../data/co2/co2_year.csv', index_col='Year').CO2
    co2_month = pd.read_csv('./../data/co2/co2_month.csv', index_col=['Year', 'Month']).CO2

    dir = CURRENT_DIR + "./../data/inversions/"
    fname = "fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
    ds = xr.open_dataset(dir + fname)

    lat = ds.latitude.values
    lon = ds.longitude.values
    time = ds.time.values

    # Set up xr.Dataset with same shape as ds but only has values 1.
    land_vals = ds.Terrestrial_flux.values
    ocean_vals = ds.Ocean_flux.values
    land_vals[np.where(land_vals != 0)] = 1 * 365/30
    ocean_vals[np.where(ocean_vals != 0)] = 1 * 365/30

    # Some grid points may have overlap of land and ocean when summed later,
    # so will produce values of 2. Set the overlap points to 0.
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

    # Basic test result array to avoid calling latitudinal_splits multiple times.
    basic_test_result = test_ds.latitudinal_splits()

    # Output dataframe.
    output_dir = CURRENT_DIR + './../output/inversions/spatial/output_all/CAMS/'
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
    """ Test that the earth_area_grid function returns the same value as the
    entire surface area of the globe.
    """

    earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)
    expected_result = test_ds.earth_area_grid(lat,lon).sum()

    assert abs(earth_surface_area - expected_result) < 1

def test_spatial_sum():
    """ Check that sum of fluxes in test_ds equals the entire area of earth (or
    of all grids created from earth_area_grid func).
    """

    def differences(dataset):
        one_month_result = dataset.sel(time = "1993-01")

        earth_surface_area = 4 * np.pi * (test_ds.earth_radius ** 2)

        total_flux = (one_month_result.Earth_Land.values +
                      one_month_result.Earth_Ocean.values).sum()

        expected_result = test_ds.earth_area_grid(lat,lon).sum() * 1e-15

        return abs(total_flux - expected_result)

    assert differences(basic_test_result) < 1
    assert differences(test_ds.latitudinal_splits(23)) < 1

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
                ("2000", "Earth_Land"),
                ("1995", "South_Land"),
                ("2006", "North_Land"),
                ("1986", "Tropical_Land"),

           ]

    for arg in args:
        assert np.subtract(*sums(*arg)) == 0

def test_cascading_window_trend_year_co2():
    df = invf.Analysis(year_output)

    start = int(str(df.data.time[0].values)[:4])
    end = int(str(df.data.time[-1].values)[:4])

    dfco2 = co2_year.loc[start:end].values

    df.data = xr.Dataset(
        {'Earth_Land': (('time'), dfco2)},
        coords={
                'time': (('time'), df.data.time.values)
               }
    )

    def cwt(window_size):
        test_df = df.cascading_window_trend(indep='co2',
                                            window_size=window_size)
        return test_df

    assert np.all(cwt(10).values.squeeze() == np.ones((1, len(df.data.time.values) - 10)))
    assert np.all(cwt(25).values.squeeze() == np.ones((1, len(df.data.time.values) - 25)))

def test_cascading_window_trend_month_co2():
    df = invf.Analysis(month_output)

    start = int(str(df.data.time[0].values)[:4])
    end = int(str(df.data.time[-1].values)[:4])

    dfco2 = co2_month.loc[start:end].values

    df.data = xr.Dataset(
        {'Earth_Land': (('time'), dfco2)},
        coords={
                'time': (('time'), df.data.time.values)
               }
    )

    def cwt(window_size):
        test_df = df.cascading_window_trend(indep='co2',
                                            window_size=window_size)
        return test_df

    assert np.all(cwt(10).values.squeeze() == np.ones((1, len(df.data.time.values) - 10*12)))
    assert np.all(cwt(25).values.squeeze() == np.ones((1, len(df.data.time.values) - 25*12)))
