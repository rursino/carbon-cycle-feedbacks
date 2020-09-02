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
from core import trendy_flux as TRENDYf

import os


""" FUNCTIONS """
def deseasonalise_instance(instance_dict):
    d_instance_dict = deepcopy(instance_dict)
    for model in d_instance_dict:
        d_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].deseasonalise(key)) for
            key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',]},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return d_instance_dict

def bandpass_instance(instance_dict, fc):
    bp_instance_dict = deepcopy(instance_dict)

    for model in bp_instance_dict:
        bp_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].bandpass(key, fc)) for
            key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return bp_instance_dict


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"

year_S1_trendy = {}
year_S3_trendy = {}
month_S1_trendy = {}
month_S3_trendy = {}
for model in os.listdir(SPATIAL_DIRECTORY):
    model_dir = SPATIAL_DIRECTORY + model + '/'

    if 'S1' in model:
        year_S1_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
        try:
            month_S1_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'month.nc'))
        except FileNotFoundError:
            pass
    elif 'S3' in model:
        year_S3_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
        try:
            month_S3_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'month.nc'))
        except FileNotFoundError:
            pass

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]


""" FIGURES """
def trendy_yearplots(simulation, save=False):
    if simulation == 'S1':
        trendy_year = year_S1_trendy
    elif simulation == 'S3':
        trendy_year = year_S3_trendy

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    for model in trendy_year:
        df = trendy_year[model].data.Earth_Land
        ax.plot(df.time, df.values)
    plt.legend(trendy_year.keys())

    ax.set_title(f"Uptake: TRENDY {simulation} - Annual", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy{simulation}_yearplots.png")

def trendy_monthplots(simulation, save=False):
    if simulation == 'S1':
        trendy_month = month_S1_trendy
    elif simulation == 'S3':
        trendy_month = month_S3_trendy

    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(111)

    for model in trendy_month:
        df = trendy_month[model].data.Earth_Land
        ax.plot(df.time, df.values)
    plt.legend(trendy_month.keys())

    ax.set_title(f"Uptake: TRENDY {simulation} - Monthly", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy{simulation}_monthplots.png")

def trendy_year_cwt(simulation, save=False, stat_values=False):
    if simulation == 'S1':
        trendy_year = year_S1_trendy
    elif simulation == 'S3':
        trendy_year = year_S3_trendy

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    index, vals = [], []
    for model in trendy_year:
        df = trendy_year[model].cascading_window_trend(variable = 'Earth_Land', indep="CO2", window_size=10)
        index.append(df.index)
        vals.append(df.values.squeeze())

    dataframe = {}
    for ind, val in zip(index, vals):
        for i, v in zip(ind, val):
            if i in dataframe.keys():
                dataframe[i].append(v)
            else:
                dataframe[i] = [v]

    df = pd.DataFrame(dataframe).T.sort_index()
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

    ax.set_title(f"Cascading Window 10-Year Trend\nTRENDY {simulation}",
                 fontsize=24, pad=20)
    ax.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    ax.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy{simulation}_year_cwt.png")

    if stat_values:
        return regstats

def trendy_month_cwt(simulation, filter, fc=None, save=False, stat_values=False):
    if simulation == 'S1':
        trendy_month = month_S1_trendy
    elif simulation == 'S3':
        trendy_month = month_S3_trendy

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    if filter == bandpass_instance:
        filter_month_trendy = filter(trendy_month, fc)
    elif filter == deseasonalise_instance:
        filter_month_trendy = filter(trendy_month)

    index, vals = [], []
    for model in filter_month_trendy:
        df = filter_month_trendy[model].cascading_window_trend(variable = "Earth_Land", indep="CO2", window_size=10)
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
        dataframe[ind] = np.pad(row, (0, 8 - len(row)), 'constant', constant_values=np.nan)

    df = pd.DataFrame(dataframe).T.sort_index()
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

    ax.set_title(f"Cascading Window 10-Year Trend\nTRENDY {simulation}", fontsize=32, pad=20)
    ax.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    ax.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "trendy_month_cwt.png")

    if stat_values:
        return regstats


""" EXECUTION """
trendy_yearplots('S1', save=False)
trendy_yearplots('S3', save=False)

trendy_monthplots('S1', save=False) #??
trendy_monthplots('S3', save=False) #??

trendy_year_cwt('S1', save=False, stat_values=True)
trendy_year_cwt('S3', save=False, stat_values=True)

trendy_month_cwt('S3', deseasonalise_instance, save=False, stat_values=True)
trendy_month_cwt('S1', bandpass_instance, fc=1/(25*12), save=True, stat_values=True)
