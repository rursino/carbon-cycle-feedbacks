""" Development of the decadal window of the airborne fraction, using the
decadal fb parameters from the FeedbackAnalysis class.
"""

""" IMPORTS """
from core import FeedbackAnalysis
import numpy as np
import pandas as pd
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
land_df = FeedbackAnalysis.INVF(co2['year'], temp['year'], invf_uptake['year'], 'Earth_Land')
ocean_df = FeedbackAnalysis.INVF(co2['year'], temp['year'], invf_uptake['year'], 'Earth_Ocean')

emission_rate = 2


land = land_df.params()
ocean = ocean_df.params()
beta = (land['beta'] + ocean['beta'])
u_gamma = (land['u_gamma'] + ocean['u_gamma'])


beta + u_gamma

b = 1 / np.log(1 + emission_rate / 100)
u = 1 - b * (beta + u_gamma)

af = 1 / u
af_results = {'mean': af.mean(axis=1), 'std': af.std(axis=1)}

af_results

plt.bar(af_results['mean'].index,
        af_results['mean'].values,
        # yerr=1.645*af_results['std'].values,
        width=5
       )

plt.bar(alpha.mean(axis=1).index - 3, alpha.mean(axis=1).values, width=3)


def plot():
    alpha = beta + u_gamma
    plt.bar(alpha.mean(axis=1).index - 3, alpha.mean(axis=1).values, width=3)
    plt.bar(af_results['mean'].index,
            af_results['mean'].values,
            # yerr=1.645*af_results['std'].values,
            width=3
           )
    plt.legend(['alpha', 'af'])

plot()



alpha = np.linspace(-2, 2)
u = 1 - (1/0.02)*alpha
af = 1 / u
plt.plot(alpha,af)
