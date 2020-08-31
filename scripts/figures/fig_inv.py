""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr
from copy import deepcopy

import seaborn as sns
sns.set_style('darkgrid')

import sys
from core import inv_flux as invf

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

dmonth_invf = deepcopy(month_invf)
for model in dmonth_invf:
    dmonth_invf[model].data = xr.Dataset(
        {key: (('time'), month_invf[model].deseasonalise(key)) for
        key in ['Earth_Land', 'Earth_Ocean', 'South_Land', 'South_Ocean',
        'North_Land', 'North_Ocean', 'Tropical_Land', 'Tropical_Ocean']},
        coords={'time': (('time'), month_invf[model].data.time.values)}
    )

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]

""" FIGURES """
def inv_yearplots(save=False):
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

def inv_year_cwt(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['Earth_Land', 'Earth_Ocean'])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var in zip_list:
        ax = fig.add_subplot(subplot)

        index, vals = [], []
        for model in year_invf:
            df = year_invf[model].cascading_window_trend(variable = var, indep="CO2", window_size=10)
            index.append(df.index)
            vals.append(df.values.squeeze())

        dataframe = {}
        for ind, val in zip(index, vals):
            for i, v in zip(ind, val):
                if i in dataframe.keys():
                    dataframe[i].append(v)
                else:
                    dataframe[i] = [v]
        for ind in dataframe:
            row = np.array(dataframe[ind])
            dataframe[ind] = np.pad(row, (0, 6 - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index().iloc[3:]
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        ax.plot(x, y, color='red')
        ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(x, x*slope + intercept)
        text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)

        stat_vals[var] = regstats

    axl.set_title("Cascading Window 10-Year Trend", fontsize=32, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_year_cwt.png")

    if stat_values:
        return stat_vals

def inv_month_cwt(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['Earth_Land', 'Earth_Ocean'])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var in zip_list:
        ax = fig.add_subplot(subplot)

        index, vals = [], []
        for model in dmonth_invf:
            df = dmonth_invf[model].cascading_window_trend(variable = var, indep="CO2", window_size=10)
            index.append(df.index)
            vals.append(df.values.squeeze())

        dataframe = {}
        for ind, val in zip(index, vals):
            for i, v in zip(ind, val):
                if i in dataframe.keys():
                    dataframe[i].append(v)
                else:
                    dataframe[i] = [v]
        for ind in dataframe:
            row = np.array(dataframe[ind])
            dataframe[ind] = np.pad(row, (0, 12 - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index().iloc[3:]
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        ax.plot(x, y, color='red')
        ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(x, x*slope + intercept)
        text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)

        stat_vals[var] = regstats

    axl.set_title("Cascading Window 10-Year Trend", fontsize=32, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_month_cwt.png")

    if stat_values:
        return stat_vals

""" EXECUTION """
inv_yearplots(save=False)

inv_monthplots(save=False)

inv_year_cwt(save=False, stat_values=True)

inv_month_cwt(save=False, stat_values=True)
