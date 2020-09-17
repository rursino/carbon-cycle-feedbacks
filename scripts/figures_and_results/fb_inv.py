""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels import api as sm
from scipy import signal
from itertools import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import os
from core import FeedbackAnalysis

import fb_input_data as fb_id


""" FUNCTIONS """
def feedback_regression(timeres, variable):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']
    reg_models = {}
    model_stats = {
        'r_squared': [],
        't_values_beta': [],
        't_values_gamma': [],
        'p_values_beta': [],
        'p_values_gamma': [],
        'mse_total': [],
        'nobs': []
    }

    model_names = uptake.keys()
    for model_name in model_names:
        U = uptake[model_name]

        start = int(U.time.values[0].astype('str')[:4])
        end = int(U.time.values[-1].astype('str')[:4])
        C = fb_id.co2[timeres].loc[start:end]
        T = fb_id.temp[timeres].sel(time=slice(str(start), str(end)))

        df = pd.DataFrame(data = {
                                "C": C,
                                "U": U[variable],
                                "T": T['Earth']
                                }
                               )

        X = sm.add_constant(df[["C", "T"]])
        Y = df["U"]

        reg_model = sm.OLS(Y, X).fit()

        reg_models[model_name] = reg_model.params[['C', 'T']].values
        model_stats['r_squared'].append(reg_model.rsquared)
        model_stats['t_values_beta'].append(reg_model.tvalues.loc['C'])
        model_stats['t_values_gamma'].append(reg_model.tvalues.loc['T'])
        model_stats['p_values_beta'].append(reg_model.pvalues.loc['C'])
        model_stats['p_values_gamma'].append(reg_model.pvalues.loc['T'])
        model_stats['mse_total'].append(reg_model.mse_total)
        model_stats['nobs'].append(reg_model.nobs)

    reg_df = pd.DataFrame(reg_models, index=['beta', 'gamma'])
    reg_df.loc['beta'] /= 2.12
    reg_df.loc['gamma'] *= fb_id.phi / fb_id.rho

    stats_df = pd.DataFrame(model_stats, index=model_names)

    return reg_df, stats_df

def all_regstat(timeres, variable):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']

    df = FeedbackAnalysis.INVF(
                                fb_id.co2[timeres],
                                fb_id.temp[timeres],
                                uptake,
                                variable
                                )

    all_regstats = {model : pd.DataFrame(df.regstats()[model])
                    for model in uptake
               }

    return all_regstats

def mean_regstat(timeres, variable):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']

    df = FeedbackAnalysis.INVF(
                                fb_id.co2[timeres],
                                fb_id.temp[timeres],
                                uptake,
                                variable
                                )

    regstat = [df.regstats()[model] for model in uptake]

    index = regstat[0].index
    columns = regstat[0].columns
    array = np.zeros([len(index), len(columns)])
    for i, j in product(range(len(index)), range(len(columns))):
        values = []
        for model_regstat in regstat:
            val = model_regstat.iloc[i,j]
            if val != np.nan:
                values.append(val)
        array[i][j] = np.nanmean(values)

    mean_regstat = pd.DataFrame(array)
    mean_regstat.index = index
    mean_regstat.columns = columns

    return mean_regstat


""" FIGURES """
def fb_inv(timeres, save=False):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']
    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ylim = []
    for subplot, region in zip(["211", "212"], ["Land", "Ocean"]):
        df = FeedbackAnalysis.INVF(
                                    fb_id.co2[timeres],
                                    fb_id.temp[timeres],
                                    uptake,
                                    "Earth_" + region
                                    )
        beta = df.params()['beta']
        gamma = df.params()['u_gamma'] # Note u_gamma is used to compare with beta.

        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)

        ylim.append([min(beta_mean.min(), gamma_mean.min()),
                     max(beta_mean.max(), gamma_mean.max())
                    ])

        ax[subplot] = fig.add_subplot(subplot)

        bar_width = 3
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

        ax[subplot].legend([r'$\beta$', r'$u_{\gamma}$'])
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    yunits = 'yr' if timeres == 'year' else timeres
    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(f'Feedback parameter   ({yunits}' + '$^{-1}$)',
                   fontsize=16,
                   labelpad=40
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_inv_{timeres}.png")

def fb_regional_inv(timeres, save=False):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']

    all_subplots = ['22'+str(i) for i in range(1,5)]
    # Note u_gamma is used to compare with beta.
    parameters = [['beta', '_Land'], ['beta', '_Ocean'],
                  ['u_gamma', '_Land'], ['u_gamma', '_Ocean']]

    vars = ['Earth', 'South', 'Tropical', 'North']
    colors = ['gray', 'blue', 'orange', 'green']
    bar_width = 1.25
    bar_position = np.arange(-2, 2) * bar_width

    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)


    for subplot, parameter in zip(all_subplots, parameters):
        for var, color, bar_pos in zip(vars, colors, bar_position):
            df = FeedbackAnalysis.INVF(
                                        fb_id.co2[timeres],
                                        fb_id.temp[timeres],
                                        uptake,
                                        var + parameter[1]
                                        )
            param = df.params()[parameter[0]]
            param_mean = param.mean(axis=1)
            param_std = param.std(axis=1)

            ax[subplot] = fig.add_subplot(subplot)

            ax[subplot].bar(param.index + 5 + bar_pos,
                            param_mean,
                            yerr=2*param_std,
                            width=bar_width,
                            color=color)

    ax["221"].set_xlabel("Land", fontsize=14, labelpad=5)
    ax["221"].xaxis.set_label_position('top')
    ax["222"].set_xlabel("Ocean", fontsize=14, labelpad=5)
    ax["222"].xaxis.set_label_position('top')
    for subplot, region in zip(["221", "223"],
                               [r'$\beta$', r'$u_{\gamma}$']):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    yunits = 'yr' if timeres == 'year' else timeres
    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(f'Feedback parameter   ({yunits}' + '$^{-1}$)',
                   fontsize=16,
                   labelpad=40
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_inv_regional_{timeres}.png")

def fb_regional_inv2(timeres, save=False):
    uptake = fb_id.invf_duptake if timeres == "month" else fb_id.invf_uptake['year']

    all_subplots = ['42'+str(i) for i in range(1,9)]
    vars = product(['Earth', 'South', 'Tropical', 'North'], ['_Land', '_Ocean'])

    # Note u_gamma is used to compare with beta.
    parameters = ['beta', 'u_gamma']
    colors = ['green', 'red']
    bar_width = 1.5
    bar_position = np.arange(-1, 1) * bar_width

    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ymin_land, ymax_land = [], []
    ymin_ocean, ymax_ocean = [], []
    for subplot, var in zip(all_subplots, vars):
        df = FeedbackAnalysis.INVF(
                                    fb_id.co2[timeres],
                                    fb_id.temp[timeres],
                                    uptake,
                                    ''.join(var)
                                    )
        params = df.params()
        for parameter, color, bar_pos in zip(parameters, colors, bar_position):
            param = params[parameter]
            param_mean = param.mean(axis=1)
            param_std = param.std(axis=1)

            ax[subplot] = fig.add_subplot(subplot)

            ax[subplot].bar(param.index + 5 + bar_pos,
                            param_mean,
                            yerr=2*param_std,
                            width=bar_width,
                            color=color)

            if var[1] == '_Land' and subplot != "421":
                ymin_land.append((param_mean - 2*param_std).min())
                ymax_land.append((param_mean + 2*param_std).max())
            if var[1] == '_Ocean' and subplot != "422":
                ymin_ocean.append((param_mean - 2*param_std).min())
                ymax_ocean.append((param_mean + 2*param_std).max())

    delta = 0.05
    for subplot in ["423", "425", "427"]:
        ax[subplot].set_ylim([min(ymin_land) - delta * abs(min(ymin_land)),
                              max(ymax_land) + delta * abs(max(ymax_land))])
    for subplot in ["424", "426", "428"]:
        ax[subplot].set_ylim([min(ymin_ocean) - delta * abs(min(ymin_ocean)),
                              max(ymax_ocean) + delta * abs(max(ymax_ocean))])

    ax["421"].set_xlabel("Land", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("Ocean", fontsize=14, labelpad=5)
    ax["422"].xaxis.set_label_position('top')
    for subplot, region in zip(["421", "423", "425", "427"],
                               ['Earth', 'South', 'Tropical', 'North']):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    yunits = 'yr' if timeres == 'year' else timeres
    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(f'Feedback parameter   ({yunits}' + '$^{-1}$)',
                   fontsize=16,
                   labelpad=40
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_inv_regional_{timeres}2.png")


""" EXECUTION """
reg_models, model_stats = feedback_regression('month', 'North_Land')
reg_models

all_regstat("month", "Earth_Land")['CAMS']
mean_regstat("year", "Earth_Land")
mean_regstat("month", "Earth_Land")

fb_inv('year', save=False)
fb_regional_inv('year')
