""" pytest: FeedbackAnalysis module.
"""


""" IMPORTS """
from core import FeedbackAnalysis

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
    global uptake, co2, temp, invf_uptake, invf_uptake_mock, trendy_uptake

    INV_DIRECTORY = "./../output/inversions/spatial/output_all/"

    co2 = pd.read_csv('./../data/co2/co2_month.csv', index_col=['Year', 'Month']).CO2
    temp = xr.open_dataset('./../output/TEMP/spatial/output_all/HadCRUT/month.nc')

    trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
    trendy_uptake = {
        model_name : xr.open_dataset(
            f'./../output/TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc'
            )
            for model_name in trendy_models if model_name != "LPJ-GUESS"
        }

    invf_models = os.listdir(INV_DIRECTORY)

    invf_uptake = {'month': {}}
    for timeres in invf_uptake:
        for model in invf_models:
            model_dir = INV_DIRECTORY + model + '/'
            invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

    invf_uptake_mock = {'month': {}}
    for timeres in invf_uptake_mock:
        for model in invf_models:
            model_dir = INV_DIRECTORY + model + '/'
            ds =  xr.open_dataset(model_dir + f'{timeres}.nc')
            start = str(ds.time.values[0])[:4]
            end = str(ds.time.values[-1])[:4]
            C = co2.loc[int(start):int(end)].values
            T = temp.sel(time=slice(start, end)).Earth.values
            invf_uptake_mock[timeres][model] = xr.Dataset(
                {'Earth_Land': (('time'), C + T)},
                coords={
                        'time': (('time'), ds.time.values)
                       }
            )

""" TESTS """
def test_whole_feedback():
    start, end = 1960, 2017
    C = co2.loc[start:end]
    T = temp.sel(time=slice(str(start), str(end))).Earth
    U = xr.DataArray(C.values + T.values,
                     dims=('time'),
                     coords={'time': T.time.values}
                    )

    df = pd.DataFrame(data = {"C": C, "U": U, "T": T})

    X = sm.add_constant(df[["C", "T"]])
    Y = df["U"]

    reg_model = sm.OLS(Y, X).fit()

    assert reg_model.params.loc['C'] == pytest.approx(1.0)
    assert reg_model.params.loc['T'] == pytest.approx(1.0)
    assert reg_model.rsquared == 1.0
    assert reg_model.pvalues.loc['C'] == 0.0
    assert reg_model.pvalues.loc['T'] == 0.0
    assert reg_model.nobs == len(C)

def test_invf_feedback_model():
    df = FeedbackAnalysis.INVF(co2, temp, invf_uptake_mock['month'], 'Earth_Land')

    params_df = df.params()
    flatten_params = (params_df['beta'].values.flatten() * 2.12)
    for val in flatten_params[~np.isnan(flatten_params)]:
        assert (val == pytest.approx(1.0))
    flatten_params = (params_df['gamma'].values.flatten())
    for val in flatten_params[~np.isnan(flatten_params)]:
        assert (val == pytest.approx(1.0))

    regstats = df.regstats()
    for model in regstats:
        rsquared = np.unique(regstats[model]['r_squared'])
        pbeta = np.unique(regstats[model]['p_values_beta'])
        pgamma = np.unique(regstats[model]['p_values_gamma'])
        nobs = np.unique(regstats[model]['nobs'])

        assert rsquared[~np.isnan(rsquared)].squeeze() == 1.0
        assert pbeta[~np.isnan(pbeta)].squeeze() == 0.0
        assert pgamma[~np.isnan(pgamma)].squeeze() == 0.0
        assert nobs[~np.isnan(nobs)].squeeze() == 120.0

def test_invf_regionals_add_to_global():
    variables = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land',
                 'Earth_Ocean', 'South_Ocean', 'Tropical_Ocean', 'North_Ocean']
    fa = {}
    for var in variables:
        fa[var] = (FeedbackAnalysis
                    .INVF(
                        co2,
                        temp,
                        invf_uptake['month'],
                        var
                        )
                    .params()
                  )


    def check_land_sum(param, model, year):
        land = {var:fa[var][param][model][year] for var in variables[:4]}

        return (land['Earth_Land']
                - land['South_Land']
                - land['Tropical_Land']
                - land['North_Land'])

    def check_ocean_sum(param, model, year):
        ocean = {var:fa[var][param][model][year] for var in variables[4:]}

        return (ocean['Earth_Ocean']
                - ocean['South_Ocean']
                - ocean['Tropical_Ocean']
                - ocean['North_Ocean'])

    zip_list = product(
        ['beta', 'gamma'], invf_uptake['month'].keys(),
        [1980, 1990, 2000, 2008]
    )
    for parameter, model_name, window in zip_list:
        ls = check_land_sum(parameter, model_name, window)
        os = check_ocean_sum(parameter, model_name, window)

        assert ((ls == pytest.approx(0.0) or np.isnan(ls)) and
                (os == pytest.approx(0.0) or np.isnan(os)))

def test_trendy_regionals_add_to_global():
    variables = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']
    fa = {}
    for var in variables:
        fa[var] = (FeedbackAnalysis
                    .TRENDY(
                        co2,
                        temp,
                        trendy_uptake,
                        var
                        )
                    .params()
                  )


    def check_sum(param, model, year):
        land = {var:fa[var][param][model][year] for var in variables}

        return (land['Earth_Land']
                - land['South_Land']
                - land['Tropical_Land']
                - land['North_Land'])

    zip_list = product(
        ['beta', 'gamma'], trendy_uptake.keys(),
        [1980, 1990, 2000, 2008]
    )
    for parameter, model_name, window in zip_list:
        assert check_sum(parameter, model_name, window) == pytest.approx(0.0)
