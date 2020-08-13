""" Regress uptake against CO2 and temp and time.
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
LAND_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'CRUTEM/year.nc')
OCEAN_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'HadSST/year.nc')
CO2_YEAR = pd.read_csv(CO2_FILE, index_col='Year')


""" CALCULATIONS """
atm = GCP['atmospheric growth']
land, ocean = GCP['land sink'], GCP['ocean sink']
fossil, LUC = GCP['fossil fuel and industry'], GCP['land-use change emissions']

tempLand = LAND_TEMP.sel(time=slice('1959', '2018')).Earth_Land
tempOcean = OCEAN_TEMP.sel(time=slice('1959', '2018')).Earth_Ocean

co2 = CO2_YEAR[2:].CO2


df = pd.DataFrame(data =
    {
        "land": land,
        "ocean": ocean,
        "CO2": co2,
        "temp": tempLand,
        "time": GCP.index
    },
    index= GCP.index)

def regression_model(df, var):
    X = sm.add_constant(df.iloc[:,2:])
    Y = df[var]
    model = sm.OLS(Y, X).fit()

    Yreg = (X * model.params.values).sum(axis=1)

    return model, Yreg

df.head()


df[['land', 'ocean', 'CO2']]

model, Yreg = regression_model(df, 'land')
plt.plot(co2, df['land'])
plt.plot(co2, Yreg)

model.params
model.params.const + model.params.CO2 * co2
