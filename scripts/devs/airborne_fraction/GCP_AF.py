""" Use the GCP fluxes to calculate the feedback-induced airborne fraction.
Note that feedback calculations are also calculated here.
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

atm = GCP['atmospheric growth']
land, ocean = GCP['land sink'], GCP['ocean sink']
fossil, LUC = GCP['fossil fuel and industry'], GCP['land-use change emissions']

temp = TEMP_DF.sel(time=slice('1959', '2018')).Earth
co2 = CO2_YEAR[2:].CO2


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

""" AIRBORNE FRACTION - DIRECT """
af = 1 - ( (-land + -ocean) / (fossil + LUC) )
af.mean()
af.std()

""" AIRBORNE FRACTION - FEEDBACK """
pdf = params_df(GCP.index, land, ocean, co2, temp)
pdf

beta = pdf.loc['beta']
gamma = pdf.loc['gamma']
u_gamma = gamma * 0.015 / 2.06

regBeta = beta.sum() * co2
regGamma = u_gamma.sum() * temp

af_feedback = 1 - ( (-regBeta + -regGamma) / (fossil + LUC) )
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


paramsT = params_df(GCP.index, land, ocean, co2 * 2.12, temp).sum(axis=1)
paramsT

def af_feedbacks(params, phi=0.015, rho=1.94):
    u = 1 + 60 * (params.beta + params.gamma * phi / rho)
    return 1 / u

af_feedbacks(paramsT)


def new_af_feedbacks(params, co2=co2, phi=0.015, rho=1.94):

    alpha = params.beta + params.gamma * phi / rho
    CO2.iloc[-2]

new_af_feedbacks(paramsT)
