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

import fig_input_data as id

import pickle

from importlib import reload
reload(TRENDYf);
reload(id);


""" FIGURES """
def trendy_yearplots(save=False):

    year_mean = id.year_mean

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    trendy_df_models = {}

    for sim in year_mean.keys():
        df = year_mean[sim].data.Earth_Land
        ax.plot(df.time, df.values)

        trendy_df_models[sim] = df

    ax.axhline(ls='--', color='k', alpha=0.5, lw=1)
    plt.legend(year_mean.keys(), fontsize=14)

    ax.set_title(f"Uptake: TRENDY - Annual", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_yearplots.png")

    return trendy_df_models

def trendy_year_cwt(save=False, stat_values=False):
    trendy_year = {
        'S1': id.year_S1_trendy,
        'S3': id.year_S3_trendy
    }

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    trendy_cwt = {}

    subplots = ['211', '212']
    stat_vals = []
    for sim, subplot in zip(trendy_year, subplots):
        trendy_year_sim = trendy_year[sim]

        ax = fig.add_subplot(subplot)
        index, vals = [], []
        for model in trendy_year_sim:
            df = trendy_year_sim[model].cascading_window_trend(
                                            variable = 'Earth_Land',
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
        df *= 1e3 # Units: MtC / yr / GtC
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        trendy_cwt[sim] = (pd
                            .DataFrame({'Year': x, 'CWT (1/yr)': y, 'std': std})
                            .set_index('Year')
                        )

        ax.plot(x, y, color='red')
        ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        # ax.plot(x, x*slope + intercept)
        text = (f"Slope: {slope:.3f} MtC.yr$^{'{-1}'}$.GtC$^{'{-1}'}$.yr$^{'{-1}'}$\n"
                f"r = {rvalue:.3f}")
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)
        ax.set_ylabel(f"{sim}", fontsize=16,
                        labelpad=5)

        stat_vals.append(regstats)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (MtC.yr$^{-1}$.GtC$^{-1}$)", fontsize=16,
                    labelpad=40)

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_year_cwt.png")

    if stat_values:
        return stat_vals

    return trendy_cwt

def trendy_seasonal_cwt(season, fc=None, save=False, stat_values=False):
    trendy_summer = {
        'S1': id.summer_S1_trendy,
        'S3': id.summer_S3_trendy
    }

    trendy_winter = {
        'S1': id.winter_S1_trendy,
        'S3': id.winter_S3_trendy
    }

    uptake = trendy_summer if season == 'summer' else trendy_winter

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    trendy_cwt = {}

    subplots = ['211', '212']
    stat_vals = []
    for sim, subplot in zip(uptake, subplots):
        uptake_sim = uptake[sim]

        ax = fig.add_subplot(subplot)
        index, vals = [], []
        for model in uptake_sim:
            df = uptake_sim[model].cascading_window_trend(
                                            variable = 'Earth_Land',
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
            dataframe[ind] = np.pad(row, (0, 5 - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index()
        df *= 1e3 # Units: MtC / yr / GtC
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        trendy_cwt[sim] = (pd
                            .DataFrame({'Year': x, 'CWT (1/yr)': y, 'std': std})
                            .set_index('Year')
                        )

        ax.plot(x, y, color='red')
        ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        # ax.plot(x, x*slope + intercept)
        text = (f"Slope: {slope:.3f} MtC.yr$^{'{-1}'}$.GtC$^{'{-1}'}$.yr$^{'{-1}'}$\n"
                f"r = {rvalue:.3f}")
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)
        ax.set_ylabel(f"{sim}", fontsize=16,
                        labelpad=5)

        stat_vals.append(regstats)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (MtC.yr$^{-1}$.GtC$^{-1}$)", fontsize=16,
                    labelpad=40)

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_{season}_cwt.png")

    if stat_values:
        return stat_vals

    return trendy_cwt

def trendy_bandpass_timeseries(fc, save=False):
    trendy_year = id.bandpass_instance(id.month_mean, fc)

    fig = plt.figure(figsize=(16,8))
    ax = fig.add_subplot(111)

    trendy_bp = {}

    for model in trendy_year:
        df = trendy_year[model].data.Earth_Land.sel(time=slice('1750', '2017'))
        ax.plot(df.time, df.values)

        trendy_bp[model] = df

    plt.legend(trendy_year.keys())

    ax.set_title(f"Uptake: TRENDY - Low-pass filtered: 1/{1/fc} years)", fontsize=32, pad=20)
    ax.set_xlabel("Year", fontsize=16, labelpad=10)
    ax.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + f"trendy_bandpass_timeseries.png")

    return trendy_bp

def trendy_powerspec(xlim, save=False):
    zip_list = zip(['211', '212'], ['S1', 'S3'])
    trendy_month = {
        'S1': id.month_S1_trendy,
        'S3': id.month_S3_trendy
    }

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ps = {}

    for subplot, sim in zip_list:
        ax = fig.add_subplot(subplot)

        x, y = [], []
        for model in trendy_month[sim]:
            trendy_month[sim][model].time_resolution = "M"
            psd = trendy_month[sim][model].psd('Earth_Land', fs=12)
            model_x = psd.iloc[:,0]
            model_y = psd.iloc[:,1]

            x.append(model_x)
            y.append(model_y)

        dataframe = {}
        for ind, val in zip(x, y):
            for i, v in zip(ind, val):
                if i in dataframe.keys():
                    dataframe[i].append(v)
                else:
                    dataframe[i] = [v]
        for ind in dataframe:
            row = np.array(dataframe[ind])
            dataframe[ind] = np.pad(row, (0, 8 - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index()
        X = df.index
        Y = df.mean(axis=1)

        ps[sim] = (pd
                        .DataFrame({'Period (Years)': X,
                                    'Spectral Variance': Y})
                        .set_index('Period (Years)')
                    )

        ax.semilogy(X, Y)
        ax.invert_xaxis()
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(*xlim, -1.0))

    axl.set_title("Power Spectrum: TRENDY Uptake", fontsize=32, pad=20)
    axl.set_xlabel(psd.columns[0], fontsize=16, labelpad=10)
    axl.set_ylabel(psd.columns[1], fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(id.FIGURE_DIRECTORY + "trendy_powerspec.png")

    return ps


id.enso.index = pd.date_range('1959-01', '2020-01', freq='M')
enso = id.enso.loc['1960-01':'2017-12']
uptake = id.month_S3_trendy['JSBACH_S3_nbp'].data.Earth_Land.sel(time=slice('1960-01', '2017-12'))


stats.linregress(enso.values.squeeze(), uptake.values)

""" EXECUTION """
trendy_yearplots(save=False)

trendy_year_cwt(save=False, stat_values=True)

trendy_seasonal_cwt('summer', save=False, stat_values=True)
trendy_seasonal_cwt('winter', save=False, stat_values=True)

trendy_bandpass_timeseries(1/40, save=False)

trendy_powerspec([7,0], save=False)


# PICKLE (YET TO BE UPDATED)
pickle.dump(trendy_yearplots(), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_yearplots.pik', 'wb'))
pickle.dump(trendy_year_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_year_cwt.pik', 'wb'))
pickle.dump(trendy_month_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_month_cwt.pik', 'wb'))
pickle.dump(trendy_bandpass_timeseries(1/40), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_bandpass_timeseries.pik', 'wb'))
pickle.dump(trendy_powerspec([7,0]), open(id.FIGURE_DIRECTORY+'/rayner_df/trendy_powerspec.pik', 'wb'))
