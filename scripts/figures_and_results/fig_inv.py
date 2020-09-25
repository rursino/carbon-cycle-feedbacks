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
    fig = plt.figure(figsize=(16,10))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_df = {}

    for subplot, var in zip(['211', '212'], ['Earth_Land', 'Earth_Ocean']):
        inv_df_models = {}
        ax = fig.add_subplot(subplot)
        for model in id.month_invf:
            df = id.month_invf[model].data[var]
            ax.plot(df.time, df.values)
            inv_df_models[model] = df
        plt.legend(id.month_invf.keys())

        inv_df[var] = inv_df_models

    axl.set_title("Uptake: Inversions - Monthly", fontsize=32, pad=20)
    axl.set_xlabel("Year", fontsize=16, labelpad=10)
    axl.set_ylabel("C flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "inv_monthplots.png")

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
            df = id.year_invf[model].cascading_window_trend(variable = var, indep="CO2", window_size=10)
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

        inv_cwt[var] = (pd
                            .DataFrame({'CO2 (ppm)': x, 'CWT (GtC/yr/ppm)': y, 'std': std})
                            .set_index('CO2 (ppm)')
                        )

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

    return inv_cwt

def inv_month_cwt(fc=None, save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['Earth_Land', 'Earth_Ocean'])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    inv_cwt = {}

    for subplot, var in zip_list:
        ax = fig.add_subplot(subplot)

        index, vals = [], []
        for model in id.dmonth_invf:
            df = id.dmonth_invf[model].cascading_window_trend(variable = var, indep="CO2", window_size=10)
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

        inv_cwt[var] = (pd
                            .DataFrame({'CO2 (ppm)': x, 'CWT (GtC/yr/ppm)': y, 'std': std})
                            .set_index('CO2 (ppm)')
                        )

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

inv_year_cwt(save=False, stat_values=True)

inv_month_cwt(save=False, stat_values=True)

inv_powerspec([10,0], save=False)

# PICKLE
pickle.dump(inv_yearplots(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_yearplots.pik', 'wb'))
pickle.dump(inv_monthplots(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_monthplots.pik', 'wb'))
pickle.dump(inv_year_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_year_cwt.pik', 'wb'))
pickle.dump(inv_month_cwt(), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_month_cwt.pik', 'wb'))
pickle.dump(inv_powerspec([10,0]), open(id.FIGURE_DIRECTORY+'/rayner_df/inv_powerspec.pik', 'wb'))
