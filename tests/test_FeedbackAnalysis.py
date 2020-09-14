""" pytest: FeedbackAnalysis module.
"""


""" IMPORTS """
from core import FeedbackAnalysis

import numpy as np
import xarray as xr
import pandas as pd
from statsmodels import api as sm
import os

import pytest


""" SETUP """
def setup_module(module):
    print('--------------------setup--------------------')
    global uptake, co2, temp, invf_uptake

    INV_DIRECTORY = "./../output/inversions/spatial/output_all/"

    co2 = pd.read_csv('./../data/co2/co2_month.csv', index_col=['Year', 'Month']).CO2
    temp = xr.open_dataset('./../output/TEMP/spatial/output_all/HadCRUT/month.nc')
    uptake = xr.open_dataset('./../output/TRENDY/spatial/mean_all/S3/month.nc')

    invf_models = os.listdir(INV_DIRECTORY)
    invf_uptake = {'month': {}}
    for timeres in invf_uptake:
        for model in invf_models:
            model_dir = INV_DIRECTORY + model + '/'
            ds =  xr.open_dataset(model_dir + f'{timeres}.nc')
            start = str(ds.time.values[0])[:4]
            end = str(ds.time.values[-1])[:4]
            C = co2.loc[int(start):int(end)].values
            T = temp.sel(time=slice(start, end)).Earth.values
            invf_uptake[timeres][model] = xr.Dataset(
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
    df = FeedbackAnalysis.INVF(co2, temp, invf_uptake['month'], 'Earth_Land')

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
