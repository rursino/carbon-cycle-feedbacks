""" Development of the new Feedback class for TRENDY.
"""

""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels import api as sm
from scipy import stats
from datetime import datetime
import os


""" INPUTS """
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

model_names = ['LPJ-GUESS', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in model_names},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in model_names if model_name != "LPJ-GUESS"}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in model_names},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in model_names if model_name != "LPJ-GUESS"}
    }
}

sim = 'S3'
timeres = 'month'


""" FUNCTIONS """
def preprocessing(sim, timeres, var):
    input_models = {}
    U_models = uptake[sim][timeres]
    for model_name in U_models:

        start, end = 1960, 2017
        C = co2[timeres].loc[start:end]
        T = temp[timeres].sel(time=slice(str(start), str(end)))
        U = U_models[model_name].sel(time=slice(str(start), str(end)))

        input_models[model_name] = pd.DataFrame(data = {
                                                "C": C,
                                                "U": U[var + "_Land"],
                                                "T": T[var]
                                                }
                                               )

    return input_models

def feedback_model(input_models):
    beta_dict, gamma_dict = {}, {}
    stats_dict = {}
    for model_name in input_models:
        time_periods = (
            (1960, 1969),
            (1970, 1979),
            (1980, 1989),
            (1990, 1999),
            (2000, 2009),
            (2007, 2017),
        )

        beta_dict['Year'] = [start for start, end in time_periods]
        gamma_dict['Year'] = [start for start, end in time_periods]

        params = {'beta': [], 'gamma': []}
        for start, end in time_periods:
            # Perform OLS regression on the three variables for analysis.
            X = input_models[model_name][["C", "T"]].loc[start:end]
            Y = input_models[model_name]["U"].loc[start:end]
            X = sm.add_constant(X)

            model = sm.OLS(Y, X).fit()
            params['beta'].append(model.params.loc['C'])
            params['gamma'].append(model.params.loc['T'])


        beta_dict[model_name] = params['beta']
        gamma_dict[model_name] = params['gamma']

    betaDF = pd.DataFrame(beta_dict).set_index('Year')
    gammaDF = pd.DataFrame(gamma_dict).set_index('Year')

    phi, rho = 0.015, 1.93
    betaDF /= 2.12
    gammaDF *= phi / rho

    return betaDF, gammaDF


""" EXECUTION """
betaDF, gammaDF = feedback_model(preprocessing("S3", "month", "Tropical"))

betaDF.mean(axis=1)
betaDF.std(axis=1)
gammaDF.mean(axis=1)
gammaDF.std(axis=1)

# Attributes
# self.params = getattr(self.model, 'params')
# self.r_squared = getattr(self.model, 'rsquared')
# self.tvalues = getattr(self.model, 'tvalues')
# self.pvalues = getattr(self.model, 'pvalues')
# self.mse_total = getattr(self.model, 'mse_total')
# self.nobs = getattr(self.model, 'nobs')
