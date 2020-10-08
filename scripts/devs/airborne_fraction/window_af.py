""" Development of the decadal window of the airborne fraction, using the
decadal fb parameters from the FeedbackAnalysis class.
"""

""" IMPORTS """
from core import FeedbackAnalysis, AirborneFraction
import numpy as np
import pandas as pd
from scipy import stats
import xarray as xr
import matplotlib.pyplot as plt
import os


""" INPUTS """
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'
INV_DIRECTORY = OUTPUT_DIR + "inversions/spatial/output_all/"

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc')
}

invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in trendy_models}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in trendy_models}
    }
}

phi, rho = 0.0071, 1.93


""" DEVS """
from importlib import reload
reload(AirborneFraction);

df = AirborneFraction.GCP(co2['year'], temp['year'])#, invf_uptake['year'])

df.window_af()


def plot():
    fig = plt.figure(figsize=(10,8))
    alpha = beta + u_gamma
    ax1 = fig.add_subplot(211)
    ax1.bar(alpha.mean(axis=1).index - 3, alpha.mean(axis=1).values, width=3)
    ax1.axhline(linestyle='--', alpha=0.5, color='k')
    ax1.axhline(emission_rate, linestyle='--', alpha=0.5, color='r')

    ax2 = fig.add_subplot(212)
    x = af_results['mean'].index
    y = af_results['mean'].values
    ax2.bar(x, y,
            # yerr=1.645*af_results['std'].values,
            width=3
           )
    for i, j in enumerate(y):
        ax2.text(x[i]+2, 0.1, f'{j:.2f}', color='blue', fontweight='bold')
    ax2.axhline(linestyle='--', alpha=0.5, color='k')

    no_outliers = [np.median(y) - 1.5 * stats.iqr(y), np.median(y) + 1.5 * stats.iqr(y)]
    y_max = y[(y > no_outliers[0]) & (y < no_outliers[1])]

    delta = 0.5
    ax2.set_xlim([min(x) - 5 , max(x) + 5])
    ax2.set_ylim([max(y.min() - delta, 0) , y_max.max() + delta])


plot()
