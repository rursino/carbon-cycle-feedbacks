""" IMPORTS """
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import xarray as xr

import sys
sys.path.append('./../core/')
import inv_flux as invf

import os


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/inversions/spatial/output_all/"

year_invf = {}
month_invf = {}
for model in os.listdir(SPATIAL_DIRECTORY):
    model_dir = SPATIAL_DIRECTORY + model + '/'
    year_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
    month_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'month.nc'))

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]

""" FIGURES """
def inv_yearplots(save=False):
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(16,10))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var in zip(['211', '212'], ['Earth_Land', 'Earth_Ocean']):
        ax = fig.add_subplot(subplot)
        for model in year_invf:
            df = year_invf[model].data[var]
            ax.plot(df.time, df.values)
        plt.legend(year_invf.keys())

    axl.set_title("Uptake: Inversions - Annual", fontsize=32, pad=20)
    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_yearplots.png")

def inv_monthplots(save=False):
    sns.set_style("darkgrid")

    fig = plt.figure(figsize=(16,10))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var in zip(['211', '212'], ['Earth_Land', 'Earth_Ocean']):
        ax = fig.add_subplot(subplot)
        for model in month_invf:
            df = month_invf[model].data[var]
            ax.plot(df.time, df.values)
        plt.legend(month_invf.keys())

    axl.set_title("Uptake: Inversions - Monthly", fontsize=32, pad=20)
    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_monthplots.png")


""" EXECUTION """
inv_yearplots(save=True)
inv_monthplots(save=True)
