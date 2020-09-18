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
    uptake = fb_id.trendy_duptake if timeres == "month" else fb_id.trendy_uptake
    start, end = 1960, 2017
    reg_models = {}
    model_stats = {}
    for simulation in ['S1', 'S3']:
        model_names = uptake[simulation][timeres].keys()
        if simulation == 'S1':
            temp = fb_id.temp_zero[timeres]
        elif simulation == 'S3':
            temp = fb_id.temp[timeres]

        sim_reg_models = {}
        sim_model_stats = {
                'r_squared': [],
                't_values_beta': [],
                't_values_gamma': [],
                'p_values_beta': [],
                'p_values_gamma': [],
                'mse_total': [],
                'nobs': []
            }

        for model_name in model_names:
            C = fb_id.co2[timeres].loc[start:end]
            T = temp.sel(time=slice(str(start), str(end)))
            U = uptake[simulation][timeres][model_name].sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T['Earth']
                                    }
                                   )

            X = sm.add_constant(df[["C", "T"]])
            Y = df["U"]
            reg_model = sm.OLS(Y, X).fit()
            sim_reg_models[model_name] = reg_model.params[['C', 'T']].values

            sim_model_stats['r_squared'].append(reg_model.rsquared)
            sim_model_stats['t_values_beta'].append(reg_model.tvalues.loc['C'])
            sim_model_stats['p_values_beta'].append(reg_model.pvalues.loc['C'])
            sim_model_stats['t_values_gamma'].append(reg_model.tvalues.loc['T'])
            sim_model_stats['p_values_gamma'].append(reg_model.pvalues.loc['T'])
            sim_model_stats['mse_total'].append(reg_model.mse_total)
            sim_model_stats['nobs'].append(reg_model.nobs)

        reg_models[simulation] = pd.DataFrame(sim_reg_models, index=['beta', 'gamma'])
        reg_models[simulation].loc['beta'] /= 2.12
        reg_models[simulation].loc['u_gamma'] = reg_models[simulation].loc['gamma'] * fb_id.phi / fb_id.rho

        model_stats[simulation] = pd.DataFrame(sim_model_stats, index=model_names)

    return reg_models, model_stats

def all_regstat(timeres, variable):
    uptake = fb_id.trendy_duptake if timeres == "month" else fb_id.trendy_uptake
    all_regstats = {}

    for simulation in ['S1', 'S3']:
        if simulation == 'S1':
            temp = fb_id.temp_zero[timeres]
        elif simulation == 'S3':
            temp = fb_id.temp[timeres]
        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2[timeres],
                                    temp,
                                    uptake[simulation][timeres],
                                    variable
                                    )
        all_regstats[simulation] = {model : pd.DataFrame(df.regstats()[model])
                   for model in uptake[simulation][timeres]}

    return all_regstats

def mean_regstat(timeres, variable):
    uptake = fb_id.trendy_duptake if timeres == "month" else fb_id.trendy_uptake
    mean_regstats = {}

    for simulation in ["S1", "S3"]:
        if simulation == 'S1':
            temp = fb_id.temp_zero[timeres]
        elif simulation == 'S3':
            temp = fb_id.temp[timeres]

        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2[timeres],
                                    temp,
                                    uptake[simulation][timeres],
                                    variable
                                    )

        regstat = [df.regstats()[model] for model in uptake[simulation][timeres]]

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

        mean_regstats[simulation] = mean_regstat

    return mean_regstats

""" FIGURES """
def fb_trendy(timeres, save=False):
    uptake = fb_id.trendy_duptake if timeres == "month" else fb_id.trendy_uptake
    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, simulation in zip(["211", "212"], ["S1", "S3"]):
        if simulation == 'S1':
            temp = fb_id.temp_zero[timeres]
        elif simulation == 'S3':
            temp = fb_id.temp[timeres]

        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2[timeres],
                                    temp,
                                    uptake[simulation][timeres],
                                    'Earth_Land'
                                    )

        whole_df = feedback_regression(timeres, "Earth_Land")[0][simulation].mean(axis=1)
        beta = df.params()['beta']
        gamma = df.params()['u_gamma']

        whole_beta = whole_df['beta']
        whole_gamma = whole_df['u_gamma']

        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)

        if timeres == 'month':
            whole_beta *= 12
            whole_gamma *= 12
            beta_mean *= 12
            beta_std *= 12
            gamma_mean *= 12
            gamma_std *= 12

        ax[subplot] = fig.add_subplot(subplot)

        ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=0.8)

        bar_width = 3
        if simulation == "S3":
            ax[subplot].bar(beta.index + 5 - bar_width / 2,
                            beta_mean,
                            yerr=beta_std * 1.645,
                            width=bar_width,
                            color='green',
                            label=r'$\beta$')
            ax[subplot].bar(gamma.index + 5 + bar_width / 2,
                            gamma_mean,
                            yerr=gamma_std * 1.645,
                            width=bar_width,
                            color='red',
                            label=r'$u_{\gamma}$')
            ax[subplot].axhline(y= whole_beta, ls='--', color='green', alpha=0.8, lw=1)
            ax[subplot].axhline(y= whole_gamma, ls='--', color='red', alpha=0.8, lw=1)
            ax[subplot].legend()

        elif simulation == "S1":
            ax[subplot].bar(beta.index + 5,
                            beta_mean,
                            yerr=beta_std,
                            width=bar_width,
                            color='green',
                            label=r'$\beta$')
            ax[subplot].axhline(y= whole_beta, ls='--', color='green', alpha=0.8, lw=1)
            ax[subplot].legend()

        ax[subplot].set_ylabel(simulation, fontsize=14, labelpad=5)

    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel('Feedback parameter   (yr$^{-1}$)', fontsize=16, labelpad=40)

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_trendy_{timeres}.png")

def fb_regional_trendy(timeres, save=False):
    uptake = (fb_id.trendy_duptake if timeres == "month" else
              fb_id.trendy_uptake)

    all_subplots = ['221', '222', '224']
    # Note u_gamma is used to compare with beta.
    parameters = [['beta', 'S1'], ['beta', 'S3'], ['u_gamma', 'S3']]

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
            if parameter[1] == 'S1':
                temp = fb_id.temp_zero[timeres]
            elif parameter[1] == 'S3':
                temp = fb_id.temp[timeres]

            df = FeedbackAnalysis.TRENDY(
                                        fb_id.co2[timeres],
                                        temp,
                                        uptake[parameter[1]][timeres],
                                        var + '_Land'
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

    ax["221"].set_xlabel("S1", fontsize=14, labelpad=5)
    ax["221"].xaxis.set_label_position('top')
    ax["222"].set_xlabel("S3", fontsize=14, labelpad=5)
    ax["222"].xaxis.set_label_position('top')
    for subplot, region in zip(["221", "224"],
                               [r'$\beta$', r'$u_{\gamma}$']):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    yunits = 'yr' if timeres == 'year' else timeres
    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(f'Feedback parameter   ({yunits}' + '$^{-1}$)',
                   fontsize=16,
                   labelpad=40
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_trendy_regional_{timeres}.png")

def fb_regional_trendy2(timeres, save=False):
    uptake = (fb_id.trendy_duptake if timeres == "month" else
              fb_id.trendy_uptake)

    all_subplots = ['42'+str(i) for i in range(1,9)]
    vars = product(['Earth', 'South', 'Tropical', 'North'], ['S1', 'S3'])

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
        if var[1] == 'S1':
            temp = fb_id.temp_zero[timeres]
        elif var[1] == 'S3':
            temp = fb_id.temp[timeres]
        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2[timeres],
                                    temp,
                                    uptake[var[1]][timeres],
                                    var[0] + '_Land'
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

            if var[1] == 'S1' and subplot != "421":
                ymin_land.append((param_mean - 2*param_std).min())
                ymax_land.append((param_mean + 2*param_std).max())
            if var[1] == 'S3' and subplot != "422":
                ymin_ocean.append((param_mean - 2*param_std).min())
                ymax_ocean.append((param_mean + 2*param_std).max())

    delta = 0.05
    for subplot in ["423", "425", "427"]:
        ax[subplot].set_ylim([min(ymin_land) - delta * abs(min(ymin_land)),
                              max(ymax_land) + delta * abs(max(ymax_land))])
    for subplot in ["424", "426", "428"]:
        ax[subplot].set_ylim([min(ymin_ocean) - delta * abs(min(ymin_ocean)),
                              max(ymax_ocean) + delta * abs(max(ymax_ocean))])

    ax["421"].set_xlabel("S1", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("S3", fontsize=14, labelpad=5)
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
        plt.savefig(fb_id.FIGURE_DIRECTORY + f"fb_trendy_regional_{timeres}2.png")


""" EXECUTION """
whole_param = feedback_regression('month', 'Earth_Land')
whole_param[0]['S1'].mean(axis=1)*12
whole_param[0]['S3'].mean(axis=1)*12



whole_param[1]['S1'].mean()
whole_param[1]['S3'].mean()

land[0].mean(axis=1) / ocean[0].mean(axis=1)


fb_trendy('month', save=False)
fb_trendy('year', save=False)

# all_regstat("month", "Earth_Land")['S3']['OCN']
# mean_regstat("year", "Earth_Land")['S3']
meanreg = mean_regstat("month", "Earth_Land")
meanreg['S1']
meanreg['S3']
