""" IMPORTS """
from core import AirborneFraction

from importlib import reload
reload(AirborneFraction);

import pandas as pd
import xarray as xr
import numpy as np
from scipy import signal

import os


""" INPUTS """
DIR = './../../'
OUTPUT_DIR = DIR + 'output/'
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

temp_zero = {}
for timeres in temp:
    temp_zero[timeres] = xr.Dataset(
        {key: (('time'), np.zeros(len(temp[timeres][key]))) for key in ['Earth', 'South', 'Tropical', 'North']},
        coords={'time': (('time'), temp[timeres].time)}
    )


invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}, 'month': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
    "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
    for model_name in trendy_models}
}



""" EXECUTIONS """
GCPdf = AirborneFraction.GCP(co2['year'], temp['year'])
GCPdf.airborne_fraction()

INVdf = AirborneFraction.INVF(co2['year'], temp['year'], invf_uptake['year'])
INVdf.airborne_fraction()
INVdf.airborne_fraction()['std'] * 1.645

TRENDYdf = AirborneFraction.TRENDY(co2['year'], temp['year'], trendy_uptake['year'])
TRENDYdf.airborne_fraction()
TRENDYdf.airborne_fraction()['std'] * 1.645

# Inversions individual models
def inv_af_models(emission_rate=2):
    land = INVdf._feedback_parameters('Earth_Land')
    ocean = INVdf._feedback_parameters('Earth_Ocean')

    beta = (land + ocean).loc['beta'] / 2.12
    u_gamma = (land + ocean).loc['u_gamma']

    b = 1 / np.log(1 + emission_rate / 100)
    u = 1 - b * (beta + u_gamma)

    time = {
        'CAMS': (1979, 2017),
        'JENA_s76': (1976, 2017),
        'JENA_s85': (1985, 2017),
        'CTRACKER': (2000, 2017),
        'JAMSTEC': (1996, 2017),
        'Rayner': (1992, 2012)
    }

    af = pd.DataFrame(time).T.sort_index()
    af['AF'] = (1 / u).values
    af.columns = ['start', 'end', 'AF']

    return af

inv_af_models()

inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']]['AF']
inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']].mean()['AF']
inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']].std()['AF']

inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']]['AF']
inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']].mean()['AF']
inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']].std()['AF']


# TRENDY individual models
def trendy_af_models(emission_rate=2):
    land = TRENDYdf._feedback_parameters('Earth_Land')

    ocean = GCPdf._feedback_parameters()['ocean']
    ocean.index = ['beta', 'gamma']
    ocean['beta'] /= 2.12
    ocean['u_gamma'] = ocean['gamma'] * 0.015 / 2.12 / 1.93

    params = land.add(ocean, axis=0)
    beta = params.loc['beta']
    u_gamma = params.loc['u_gamma']

    b = 1 / np.log(1 + emission_rate / 100)
    u = 1 - b * (beta + u_gamma)

    return 1 / u

trendy_af_models()
