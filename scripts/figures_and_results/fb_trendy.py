""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import os
from core import FeedbackAnalysis


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


""" SETUP """
variable = 'Earth_Land'
timeres = 'year'
sim = 'S1'


""" EXECUTION """
df = FeedbackAnalysis.TRENDY(co2[timeres], temp[timeres], uptake[sim][timeres],
                             variable)

beta, gamma = df.params()

gamma

pd.DataFrame(regstats['OCN'], index=regstats['Year'])

betaDF.mean(axis=1)
betaDF.std(axis=1)
gammaDF.mean(axis=1)
gammaDF.std(axis=1)


""" FIGURES """
def fb_trendy():
    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ylim = []
    for subplot, simulation in zip(["211", "212"], ["S1", "S3"]):
        beta = df.merge_params(simulation, 'beta') / 2.12
        gamma = df.merge_params(simulation, 'gamma') * phi / rho

        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)

        ylim.append([min(beta_mean.min(), gamma_mean.min()),
                     max(beta_mean.max(), gamma_mean.max())
                    ])

        ax[subplot] = fig.add_subplot(subplot)

        bar_width = 3
        if simulation == "S3":
            ax[subplot].bar(beta.index + 5 - bar_width / 2,
                            beta_mean,
                            yerr=beta_std,
                            width=bar_width,
                            color='green')
            ax[subplot].bar(gamma.index + 5 + bar_width / 2,
                            gamma_mean,
                            yerr=gamma_std,
                            width=bar_width,
                            color='red')
        elif simulation == "S1":
            ax[subplot].bar(beta.index + 5,
                            beta_mean,
                            yerr=beta_std,
                            width=bar_width,
                            color='green')

        ax[subplot].legend([r'$\beta$', r'$u_{\gamma}$'])
        ax[subplot].set_ylabel(simulation, fontsize=14, labelpad=5)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel('Feedback parameter   (yr$^{-1}$)', fontsize=16, labelpad=40)

    # return {f'{parameter}': merge_mean, 'std': merge_std}



""" EXECUTION """
fb_trendy()
