""" pytest: AirborneFraction module.
"""

""" IMPORTS """
from core import AirborneFraction

import numpy as np
import xarray as xr
import pandas as pd
from statsmodels import api as sm
import os
from itertools import *

import pytest


""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global co2_year, temp_year, temp_month, invf_uptake
    global invf_uptake_mock, trendy_uptake_mock

    INV_DIRECTORY = "./../output/inversions/spatial/output_all/"
    TRENDY_DIRECTORY = './../output/TRENDY/spatial/output_all/'

    co2_year = pd.read_csv("./../data/CO2/co2_year.csv", index_col=["Year"]).CO2

    temp_year = xr.open_dataset('./../output/TEMP/spatial/output_all/HadCRUT/year.nc')

    invf_models = os.listdir(INV_DIRECTORY)
    invf_uptake = {'year': {}}
    for timeres in invf_uptake:
        for model in invf_models:
            model_dir = INV_DIRECTORY + model + '/'
            invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

    invf_uptake_mock = {'year': {}}
    for timeres in invf_uptake_mock:
        for model in invf_models:
            model_dir = INV_DIRECTORY + model + '/'
            ds =  xr.open_dataset(model_dir + f'{timeres}.nc')
            start = str(ds.time.values[0])[:4]
            end = str(ds.time.values[-1])[:4]
            C = co2_year.loc[int(start):int(end)].values
            T = temp_year.sel(time=slice(start, end)).Earth.values
            invf_uptake_mock[timeres][model] = xr.Dataset(
                {key: (('time'), (C + T) / 2) for key in ['Earth_Land', 'Earth_Ocean']},
                coords={
                        'time': (('time'), ds.time.values)
                       }
            )

    trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
    trendy_uptake = {
        "year": {model_name : xr.open_dataset(TRENDY_DIRECTORY + f'{model_name}_S3_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(TRENDY_DIRECTORY + f'{model_name}_S3_nbp/month.nc')
        for model_name in trendy_models if model_name != "LPJ-GUESS"}
    }

    trendy_uptake_mock = {'year': {}}
    for timeres in trendy_uptake_mock:
        for model in trendy_models:
            model_dir = TRENDY_DIRECTORY + model + '_S3_nbp/'
            start = '1960'
            end = '2017'
            ds =  xr.open_dataset(model_dir + f'{timeres}.nc').sel(time=slice(start, end))
            C = co2_year.loc[int(start):int(end)].values
            T = temp_year.sel(time=slice(start, end)).Earth.values
            trendy_uptake_mock[timeres][model] = xr.Dataset(
                {'Earth_Land': (('time'), (C + T) / 2)},
                coords={
                        'time': (('time'), ds.time.values)
                       }
            )


""" TESTS """
def test_GCP_airborne():
    df = AirborneFraction.GCP(co2_year, temp_year)
    df.GCP['land sink'] = (df.co2 + df.temp) / 2
    df.GCP['ocean sink'] = (df.co2 + df.temp) / 2

    params = df._feedback_parameters()
    for sink, param in product(['land', 'ocean'], ['CO2', 'temp']):
        assert params[sink][param] == pytest.approx(0.5)

    phi = 0.015 / 2.12
    rho = 1.93

    emission_rate = 2
    b = 1 / np.log(1 + emission_rate / 100)
    beta = 0.5 * 2 / 2.12
    u_gamma = 0.5 * 2 * phi / rho
    u = 1 - b * (beta + u_gamma)
    test_af = 1 / u

    assert df.airborne_fraction(emission_rate) == pytest.approx(test_af)

def test_invf_airborne():
    df = AirborneFraction.INVF(co2_year, temp_year, invf_uptake_mock['year'])

    land = df._feedback_parameters('Earth_Land')
    ocean = df._feedback_parameters('Earth_Ocean')

    for region, param in product([land, ocean], ['beta', 'gamma']):
        assert region.loc[param].mean() == pytest.approx(0.5)
        assert region.loc[param].std() == pytest.approx(0.)

    phi = 0.015 / 2.12
    rho = 1.93

    emission_rate = 2
    b = 1 / np.log(1 + emission_rate / 100)
    beta = 0.5 * 2 / 2.12
    u_gamma = 0.5 * 2 * phi / rho
    u = 1 - b * (beta + u_gamma)
    test_af = 1 / u

    assert df.airborne_fraction(emission_rate)['mean'] == pytest.approx(test_af)
    assert df.airborne_fraction(emission_rate)['std'] == pytest.approx(0.)

def test_trendy_airborne():
    df = AirborneFraction.TRENDY(co2_year, temp_year, trendy_uptake_mock['year'])
    GCPdf = AirborneFraction.GCP(co2_year, temp_year)
    GCPdf.GCP['land sink'] = (GCPdf.co2 + GCPdf.temp) / 2
    GCPdf.GCP['ocean sink'] = (GCPdf.co2 + GCPdf.temp) / 2

    phi = 0.015 / 2.12
    rho = 1.93

    land = df._feedback_parameters('Earth_Land')
    ocean = GCPdf._feedback_parameters()['ocean']

    ocean.index = ['beta', 'gamma']
    ocean['beta'] /= 2.12
    ocean['u_gamma'] = ocean['gamma'] * phi / rho

    for parameter, divisor in zip(['beta', 'gamma'], [2.12, 1]):
        assert land.loc[parameter].mean() == pytest.approx(0.5 / divisor)
        assert land.loc[parameter].std() == pytest.approx(0.)
        assert ocean[parameter] == pytest.approx(0.5 / divisor)

    # Test function
    emission_rate = 2
    b = 1 / np.log(1 + emission_rate / 100)
    beta = 0.5 * 2 / 2.12
    u_gamma = 0.5 * 2 * phi / rho
    u = 1 - b * (beta + u_gamma)
    test_af = 1 / u

    # Class function
    params = land.add(ocean, axis=0)
    class_beta = params.loc['beta']
    class_u_gamma = params.loc['u_gamma']

    class_b = 1 / np.log(1 + emission_rate / 100)
    class_u = 1 - class_b * (class_beta + class_u_gamma)

    class_af = 1 / class_u

    assert class_af.mean() == pytest.approx(test_af)
