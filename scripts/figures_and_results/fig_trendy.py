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
SPATIAL_MEAN_DIRECTORY = "./../../output/TRENDY/spatial/mean_all/"

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

year_mean = {
    "S1": TRENDYf.Analysis(xr.open_dataset(SPATIAL_MEAN_DIRECTORY + 'S1/year.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(SPATIAL_MEAN_DIRECTORY + 'S3/year.nc'))
}
month_mean = {
    "S1": TRENDYf.Analysis(xr.open_dataset(SPATIAL_MEAN_DIRECTORY + 'S1/year.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(SPATIAL_MEAN_DIRECTORY + 'S3/year.nc'))
}

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]


""" FIGURES """
def trendy_yearplots(save=False):

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    for sim in year_mean.keys():
        df = year_mean[sim].Earth_Land
        ax.plot(df.time, df.values)
    ax.axhline(ls='--', color='k', alpha=0.5, lw=1)
    plt.legend(year_mean.keys(), fontsize=14)

    ax.set_title(f"Uptake: TRENDY - Annual", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy_yearplots.png")

def trendy_year_cwt(save=False, stat_values=False):
    trendy_year = {
        'S1': year_S1_trendy,
        'S3': year_S3_trendy
    }

    fig = plt.figure(figsize=(16,12))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    subplots = ['211', '212']
    stat_vals = []
    for sim, subplot in zip(trendy_year, subplots):
        trendy_year_sim = trendy_year[sim]

        ax = fig.add_subplot(subplot)
        index, vals = [], []
        for model in trendy_year_sim:
            df = trendy_year_sim[model].cascading_window_trend(
                                            variable = 'Earth_Land',
                                            indep="CO2",
                                            window_size=10)
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

        stat_vals.append(regstats)

    axl.set_title(f"Cascading Window 10-Year Trend - TRENDY",
                 fontsize=30, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy{simulation}_year_cwt.png")

    if stat_values:
        return stat_vals

def trendy_month_cwt(fc=None, save=False, stat_values=False):
    trendy_month = {
        'S1': month_S1_trendy,
        'S3': month_S3_trendy
    }

    fig = plt.figure(figsize=(16,12))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    subplots = ['211', '212']
    stat_vals = []
    for sim, subplot in zip(trendy_month, subplots):
        trendy_month_sim = trendy_month[sim]

        ax = fig.add_subplot(subplot)
        index, vals = [], []
        for model in trendy_month_sim:
            df = trendy_month_sim[model].cascading_window_trend(
                                            variable = 'Earth_Land',
                                            indep="CO2",
                                            window_size=10)
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

        stat_vals.append(regstats)

    axl.set_title(f"Cascading Window 10-Year Trend - TRENDY",
                 fontsize=30, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy{simulation}_year_cwt.png")

    if stat_values:
        return stat_vals

def trendy_bandpass_year(fc, save=False):
    trendy_year = bandpass_instance(month_mean, fc)

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    for model in trendy_year:
        df = trendy_year[model].data.Earth_Land
        ax.plot(df.time, df.values)
    plt.legend(trendy_year.keys())

    ax.set_title(f"Uptake: TRENDY - Low-pass filtered: {1/fc} years)", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy_bandpass_year.png")

def trendy_powerspec(xlim, save=False):
    zip_list = zip(['211', '212'], ['S1', 'S3'])

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, sim in zip_list:
        ax = fig.add_subplot(subplot)

        month_mean[sim].time_resolution = "M"
        psd = month_mean[sim].psd('Earth_Land', fs=12)
        x = psd.iloc[:,0]
        y = psd.iloc[:,1]

        ax.semilogy(x, y)
        ax.invert_xaxis()
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(*xlim, -1.0))

    axl.set_title("Power Spectrum: TRENDY Uptake", fontsize=32, pad=20)
    axl.set_xlabel(psd.columns[0], fontsize=16, labelpad=10)
    axl.set_ylabel(psd.columns[1], fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "trendy_powerspec.png")


""" EXECUTION """
trendy_yearplots(save=False)

trendy_year_cwt(save=False, stat_values=True)

trendy_month_cwt(save=False, stat_values=True)

trendy_bandpass_year(1/40, save=False)

trendy_powerspec([10,0], save=False)
