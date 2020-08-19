""" Feedback analysis on GCP.
"""

""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm
import matplotlib.pyplot as plt


""" INPUTS """
GCP_FILE = './../../../data/GCP/budget.csv'
TEMP_OUTPUT_DIR = './../../../output/TEMP/spatial/output_all/'
CO2_FILE = './../../../data/CO2/co2_year.csv'

GCP = pd.read_csv(GCP_FILE, index_col='Year')
TEMP_DF = xr.open_dataset(TEMP_OUTPUT_DIR + 'HadCRUT/year.nc')
CO2_YEAR = pd.read_csv(CO2_FILE, index_col='Year')


""" FUNCTIONS """
def regression_model(index, var, co2, temp):

    df = pd.DataFrame(data =
        {
            "var": var,
            "CO2": co2,
            "temp": temp
        },
        index= index)

    X = sm.add_constant(df[['CO2', 'temp']])
    Y = df['var']
    model = sm.OLS(Y, X).fit()

    return model

def params_df(index, land, ocean, co2, temp):

    return pd.DataFrame(
        {
            'land': regression_model(index, land, co2, temp).params.values,
            'ocean': regression_model(index, ocean, co2, temp).params.values
        },
        index = ['const', 'beta', 'gamma']
    )

def plot_regression(model):

    x = co2, temp
    y


""" CALCULATIONS """
atm = GCP['atmospheric growth']
land, ocean = GCP['land sink'], GCP['ocean sink']
fossil, LUC = GCP['fossil fuel and industry'], GCP['land-use change emissions']

temp = TEMP_DF.sel(time=slice('1959', '2018')).Earth
co2 = CO2_YEAR[2:].CO2

pdf = params_df(GCP.index, land, ocean, co2, temp)

beta = pdf.loc['beta']
gamma = pdf.loc['gamma']
u_gamma = gamma * 0.015 / 2.06

land_model = regression_model(GCP.index, land, co2, temp)
ocean_model = regression_model(GCP.index, ocean, co2, temp)

land_model.summary()

ocean_model.summary()

land_model.params
