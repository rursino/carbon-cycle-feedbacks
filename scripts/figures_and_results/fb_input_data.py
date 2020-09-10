""" Store all functionality to import and pre-process relevant input data for
feedback_analysis figures here."""

""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels import api as sm
from scipy import signal

import os
from core import FeedbackAnalysis


""" FUNCTIONS """
def deseasonalise(x):
    """
    """

    fs = 12
    fc = 365/667

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'low')

    return signal.filtfilt(b, a, x)


""" INPUTS """
DIR = './../../'
OUTPUT_DIR = DIR + 'output/'

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

trendy_models = ['LPJ-GUESS', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in trendy_models if model_name != "LPJ-GUESS"}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in trendy_models if model_name != "LPJ-GUESS"}
    }
}

trendy_duptake = {'S1': {'year': {}, 'month': {}}, 'S3': {'year': {}, 'month': {}}}
for sim in trendy_uptake:
    for timeres in trendy_uptake[sim]:
        for model in trendy_uptake[sim][timeres]:
            df_models = trendy_uptake[sim][timeres]
            trendy_duptake[sim][timeres][model] = xr.Dataset(
                {key: (('time'), deseasonalise(df_models[model][key].values)) for
                key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']},
                coords={'time': (('time'), df_models[model].time.values)}
            )

phi, rho = 0.015, 1.93
