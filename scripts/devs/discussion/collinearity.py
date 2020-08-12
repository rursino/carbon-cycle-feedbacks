""" An exploration into the collinearity between CO2 and tempearture when
regressing uptake to these two explanatory variables.
"""

""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt


""" INPUTS """
GCP_FILE = './../../../data/GCP/budget.csv'
TEMP_OUTPUT_DIR = './../../../output/TEMP/spatial/output_all/'
CO2_FILE = './../../../data/CO2/co2_year.csv'

GCP = pd.read_csv(GCP_FILE, index_col='Year')
LAND_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'CRUTEM/year.nc')
OCEAN_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'HadSST/year.nc')
CO2_YEAR = pd.read_csv(CO2_FILE, index_col='Year')

LAND_TEMP

co2 = CO2_YEAR.CO2
co2
landTemp = LAND_TEMP.Earth_Land.sel(time=slice('1957', '2018')).values
oceanTemp = OCEAN_TEMP.Earth_Ocean.sel(time=slice('1957', '2018')).values

stats.linregress(co2, landTemp).rvalue
stats.linregress(co2, oceanTemp).rvalue

plt.scatter(co2, landTemp)
plt.scatter(co2, oceanTemp)


plt.scatter(landTemp, oceanTemp)
