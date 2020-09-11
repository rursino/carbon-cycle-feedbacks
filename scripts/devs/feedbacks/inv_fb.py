""" Fundamental analysis of inversion FeedbackAnalysis to check with the results
from the INVF class.
"""

""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels import api as sm
from scipy import stats
from scipy import signal
import os


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
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'
INV_DIRECTORY = "./../../../output/inversions/spatial/output_all/"


invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}, 'month': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

invf_duptake = {}
for model in invf_uptake['month']:
    invf_duptake[model] = xr.Dataset(
        {key: (('time'), deseasonalise(invf_uptake['month'][model][key].values)) for
        key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',
        'Earth_Ocean', 'South_Ocean', 'North_Ocean', 'Tropical_Ocean']},
        coords={'time': (('time'), invf_uptake['month'][model].time.values)}
    )


temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}


co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

""" DEVS """
model_name = 'JENA_s76'
time_periods = [
    (1980, 1989),
    (1990, 1999),
    (2000, 2009),
    (2008, 2017),
]
# time_periods = [
#     (1980, 2017)
# ]

def feedback_window(timeres, variable):

    uptake_to_pass = invf_duptake if timeres == 'month' else invf_uptake['year']

    params = {'Year': [], 'beta': [], 'gamma': []}

    for period in time_periods:
        start, end = period
        df = pd.DataFrame(
            {
                "U": uptake_to_pass[model_name][variable].sel(time=slice(str(start), str(end))),
                "C": co2[timeres].loc[start:end].values,
                "T": temp[timeres][variable.split('_')[0]].sel(time=slice(str(start), str(end)))
            }
            )

        X = sm.add_constant(df[['C', 'T']])
        Y = df['U']

        model = sm.OLS(Y, X).fit()

        params['Year'].append(start)
        params['beta'].append(model.params.loc['C'] / 2.12)
        params['gamma'].append(model.params.loc['T'] * 0.015 / 1.93)

    return pd.DataFrame(params).set_index('Year')

feedback_window('year', 'Earth_Land')
feedback_window('month', 'Earth_Land')
