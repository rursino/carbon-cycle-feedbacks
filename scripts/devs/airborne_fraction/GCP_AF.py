""" Use the GCP fluxes to calculate the feedback-induced airborne fraction.
Note that feedback calculations are also calculated here.
"""

""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

""" INPUTS """
GCP_FILE = './../../../data/GCP/budget.csv'
TEMP_OUTPUT_DIR = './../../../output/TEMP/spatial/output_all/'
CO2_FILE = './../../../data/CO2/co2_year.csv'

GCP = pd.read_csv(GCP_FILE, index_col='Year')
LAND_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'CRUTEM/year.nc')
OCEAN_TEMP = xr.open_dataset(TEMP_OUTPUT_DIR + 'HadSST/year.nc')
CO2_YEAR = pd.read_csv(CO2_FILE, index_col='Year')

""" FEEDBACK CALCULATIONS """
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
        "temp": tempLand
    },
    index= GCP.index)

def regression_model(df, var):
    X = sm.add_constant(df[['CO2', 'temp']])
    Y = df[var]
    model = sm.OLS(Y, X).fit()

    return model

params_df = pd.DataFrame(
    {
        'land': regression_model(df, 'land').params.values,
        'ocean': regression_model(df, 'ocean').params.values
    },
    index = ['const', 'beta', 'gamma']
)

params_df

""" AIRBORNE FRACTION - DIRECT """
af = 1 - ( (-land + -ocean) / (fossil + LUC) )
af.mean()
af.std()

""" AIRBORNE FRACTION - FEEDBACK """
params_df.loc['gamma'][0]

np.dot(params_df.loc['gamma'], (tempLand, tempOcean))

regBeta = params_df.loc['beta'].sum() * co2
regGamma = np.dot(params_df.loc['gamma'], (tempLand, tempOcean))
regConst = params_df.loc['const'].sum()
af_feedback = 1 - ( (-regBeta + -regGamma + -regConst) / (fossil + LUC) )
af_feedback.mean()
af_feedback.std()

residual = af_feedback - af
residual.mean()
residual.std()


""" This method of calculating AF produces similar mean but a much smaller std.
However, there are two flaws to this method.
Firstly, airborne fraction is only truly applicable on an annual basis, but
the parameters are calculated based on a 60-year regression. So, the airborne
fraction values are an average over the 60-year period only. In fact, there
should only be one airborne fraction value for the 60-year period, but this is
difficult because there is no simple way to pick one value for temperature and
CO2.
Secondly, it is worth noting that the only difference between this method and
the direct calculation is the replacement of land and ocean uptake obs to the
regressed versions. Therefore there is only lost residual as a result of the
regressions to get the parameters.
This may essentially render this method useless.
However, we know have values of beta and gamma (and the constant).
Adding knowledge of radiative forcing, which introduces a relationship between
CO2 concentrations (or emissions) and temperature, could bring down the
uncertainty associated with airborne fraction, which is very large in this
analysis and in the general literature.
"""
