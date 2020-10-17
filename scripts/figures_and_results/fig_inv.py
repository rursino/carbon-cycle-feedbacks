""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
sns.set_style('darkgrid')
from core import inv_flux as invf

import os

import fig_input_data as id

import pickle

from importlib import reload
reload(invf);
reload(id);


""" FIGURES """
def inv_yearplots(save=False):
    fig = plt.figure(figsize=(16,10))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_df = {}

    for subplot, var in zip(['211', '212'], ['Earth_Land', 'Earth_Ocean']):
        inv_df_models = {}
        ax = fig.add_subplot(subplot)
        for model in id.year_invf:
            df = id.year_invf[model].data[var]
            ax.plot(df.time, df.values)

            inv_df_models[model] = df

        plt.legend(id.year_invf.keys())

        inv_df[var] = inv_df_models

    axl.set_title("Uptake: Inversions - Annual", fontsize=32, pad=20)
    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_yearplots.png")

    return inv_df

def inv_monthplots(save=False):
    uptake = id.month_invf
    fig = plt.figure(figsize=(16,10))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_df = {}

    for subplot, var in zip(['211', '212'], ['Earth_Land', 'Earth_Ocean']):
        inv_df_models = {}
        ax = fig.add_subplot(subplot)
        for model in uptake:
            df = uptake[model].data[var]
            ax.plot(df.time, df.values)
            inv_df_models[model] = df
        plt.legend(uptake.keys())

        inv_df[var] = inv_df_models

    axl.set_title("Uptake: Inversions - Monthly", fontsize=32, pad=20)
    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_monthplots.png")

    return inv_df

def inv_seasonalplots(season, save=False):
    uptake = id.summer_invf if season == 'summer' else id.winter_invf
    fig = plt.figure(figsize=(16,6))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_df = {}
    inv_df_models = {}
    ax = fig.add_subplot('111')
    for model in uptake:
        df = uptake[model].data['Earth_Land']
        ax.plot(df.time, df.values)
        inv_df_models[model] = df
    plt.legend(uptake.keys())

    inv_df['Earth_Land'] = inv_df_models

    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"inv_{season}plots.png")

    return inv_df

def inv_year_cwt(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['Earth_Land', 'Earth_Ocean'])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_cwt = {}

    for subplot, var in zip_list:
        ax = fig.add_subplot(subplot)

        index, vals = [], []
        for model in id.year_invf:
            df = id.year_invf[model].cascading_window_trend(variable = var,
                                                            window_size=10)
            index.append(df.index)
            vals.append(df.values)

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

        df = pd.DataFrame(dataframe).T.sort_index()
        df *= 1e3 # Units MtC / yr / GtC
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        inv_cwt[var] = (pd
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
                f"r = {rvalue:.3f}\n")
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)
        ax.set_ylabel(f"{var.split('_')[1].title()}", fontsize=16,
                        labelpad=5)

        stat_vals[var] = regstats

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (MtC.yr$^{-1}$.GtC$^{-1}$)", fontsize=16,
                    labelpad=40)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_year_cwt.png")

    if stat_values:
        return stat_vals

    return inv_cwt

def inv_seasonal_cwt(fc=None, save=False, stat_values=False):
    uptake = {
        'winter': id.winter_invf,
        'summer': id.summer_invf
    }
    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_cwt = {}

    for season, subplot in zip(uptake.keys(), ['211', '212']):
        var = 'Earth_Land'
        ax = fig.add_subplot(subplot)

        index, vals = [], []
        for model in uptake[season]:
            df = uptake[season][model].cascading_window_trend(variable = var,
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
            dataframe[ind] = np.pad(row, (0, 6 - len(row)), 'constant', constant_values=np.nan)

        df = pd.DataFrame(dataframe).T.sort_index()
        df *= 1e3 # Units MtC / yr / GtC
        x = df.index
        y = df.mean(axis=1)
        std = df.std(axis=1)

        inv_cwt[season] = (pd
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
                f"r = {rvalue:.3f}\n")
        xtext = x.min() + 0.75 * (x.max() -x.min())
        ytext = (y-2*std).min() +  0.75 * ((y+2*std).max() - (y-2*std).min())
        ax.text(xtext, ytext, text, fontsize=15)

        ax.set_ylabel(f"{season.title()}", fontsize=16,
                        labelpad=5)

        stat_vals[season] = regstats

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (MtC.yr$^{-1}$.GtC$^{-1}$)", fontsize=16,
                    labelpad=40)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"inv_seasonal_cwt.png")

    if stat_values:
        return stat_vals

    return inv_cwt

def inv_powerspec(xlim, save=False):
    zip_list = zip(['211', '212'], ['Earth_Land', 'Earth_Ocean'])

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_cwt = {}

    for subplot, var in zip_list:
        ax = fig.add_subplot(subplot)

        psd = id.month_invf['JAMSTEC'].psd(var, fs=12)
        x = psd.iloc[:,0]
        y = psd.iloc[:,1]

        inv_cwt[var] = (pd
                            .DataFrame({'Period (years)': x.values,
                                        'Spectral Variance': y.values})
                            .set_index('Period (years)')
                        )

        ax.semilogy(x, y)
        ax.invert_xaxis()
        ax.set_xlim(xlim)
        ax.set_xticks(np.arange(*xlim, -1.0))

    axl.set_title("Power Spectrum: Inversion Uptake", fontsize=32, pad=20)
    axl.set_xlabel(psd.columns[0], fontsize=16, labelpad=10)
    axl.set_ylabel(psd.columns[1], fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_powerspec.png")

    return inv_cwt


""" EXECUTION """
inv_yearplots(save=False)

inv_monthplots(save=False)

inv_seasonalplots('winter', save=False)
inv_seasonalplots('summer', save=False)

inv_year_cwt(save=False, stat_values=False)
inv_seasonal_cwt(save=False, stat_values=True)

inv_powerspec([10,0], save=False)

# PICKLE (YET TO BE UPDATED)
pickle.dump(inv_yearplots(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_yearplots.pik', 'wb'))
pickle.dump(inv_monthplots(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_monthplots.pik', 'wb'))
pickle.dump(inv_year_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_year_cwt.pik', 'wb'))
pickle.dump(inv_month_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_month_cwt.pik', 'wb'))
pickle.dump(inv_powerspec([10,0]), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_powerspec.pik', 'wb'))
