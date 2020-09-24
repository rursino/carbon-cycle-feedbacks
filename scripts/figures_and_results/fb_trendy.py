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
def feedback_regression(variable):
    uptake = fb_id.trendy_duptake
    start, end = 1960, 2017
    reg_models = {}
    model_stats = {}
    for simulation in ['S1', 'S3']:
        model_names = uptake[simulation]['month'].keys()

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
            C = fb_id.co2['month'].loc[start:end]
            T = fb_id.temp['month'].sel(time=slice(str(start), str(end)))
            if simulation == 'S1':
                U = uptake['S1']['month'][model_name]
            elif simulation == 'S3':
                U = uptake['S3']['month'][model_name] - uptake['S1']['month'][model_name]
            U = U.sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T['Earth']
                                    }
                                   )

            if simulation == 'S1':
                x_var = 'C'
                other_var = 'T'
            elif simulation == 'S3':
                x_var = 'T'
                other_var = 'C'
            X = sm.add_constant(df[x_var])
            Y = df["U"]
            reg_model = sm.OLS(Y, X).fit()
            reg_model_params = reg_model.params
            reg_model_params[other_var] = 0
            sim_reg_models[model_name] = (reg_model_params
                                            .loc[[x_var, other_var]]
                                            .sort_index()
                                         )
            sim_model_stats['r_squared'].append(reg_model.rsquared)
            if simulation == 'S1':
                sim_model_stats['t_values_beta'].append(reg_model.tvalues.loc['C'])
                sim_model_stats['p_values_beta'].append(reg_model.pvalues.loc['C'])
                sim_model_stats['t_values_gamma'].append(np.nan)
                sim_model_stats['p_values_gamma'].append(np.nan)
            elif simulation == 'S3':
                sim_model_stats['t_values_beta'].append(np.nan)
                sim_model_stats['p_values_beta'].append(np.nan)
                sim_model_stats['t_values_gamma'].append(reg_model.tvalues.loc['T'])
                sim_model_stats['p_values_gamma'].append(reg_model.pvalues.loc['T'])
            sim_model_stats['mse_total'].append(reg_model.mse_total)
            sim_model_stats['nobs'].append(reg_model.nobs)

        df = pd.DataFrame(sim_reg_models)
        df.index = ['beta', 'gamma']
        reg_models[simulation] = df
        reg_models[simulation].loc['beta'] /= 2.12
        reg_models[simulation].loc['u_gamma'] = reg_models[simulation].loc['gamma'] * fb_id.phi / fb_id.rho

        model_stats[simulation] = pd.DataFrame(sim_model_stats, index=model_names)

    return reg_models, model_stats

def all_regstat(variable):
    uptake = fb_id.trendy_duptake
    all_regstats = {}

    for simulation in ['S1', 'S3']:
        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2['month'],
                                    fb_id.temp['month'],
                                    uptake,
                                    variable
                                    )
        all_regstats[simulation] = {model : pd.DataFrame(df.regstats()[simulation][model])
                   for model in uptake[simulation]['month']}

    return all_regstats

def median_regstat(variable):
    uptake = fb_id.trendy_duptake
    mean_regstats = {}

    for simulation in ["S1", "S3"]:
        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2['month'],
                                    fb_id.temp['month'],
                                    uptake,
                                    variable
                                    )

        regstat = [df.regstats()[simulation][model] for model in uptake[simulation]['month']]

        index = regstat[0].index
        columns = regstat[0].columns
        array = np.zeros([len(index), len(columns)])
        for i, j in product(range(len(index)), range(len(columns))):
            values = []
            for model_regstat in regstat:
                val = model_regstat.iloc[i,j]
                if val != np.nan:
                    values.append(val)
            array[i][j] = np.nanmedian(values)

        mean_regstat = pd.DataFrame(array)
        mean_regstat.index = index
        mean_regstat.columns = columns

        mean_regstats[simulation] = mean_regstat

    return mean_regstats


""" FIGURES """
def fb_trendy(save=False):
    uptake = fb_id.trendy_duptake
    fig = plt.figure(figsize=(14,6))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    bar_width = 3
    param_details = {
        'beta': {
                    'bar_width': -bar_width,
                    'color': 'green',
                    'label': r'$\beta$'
                },
        'u_gamma': {
                    'bar_width': bar_width,
                    'color': 'red',
                    'label': r'$u_{\gamma}$'
                }
    }

    for simulation, param in zip(["S1", "S3"], ['beta', 'u_gamma']):
        subplot = '111'
        bw = param_details[param]['bar_width']
        color = param_details[param]['color']
        label = param_details[param]['label']

        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2['month'],
                                    fb_id.temp['month'],
                                    uptake,
                                    'Earth_Land'
                                    )

        whole_df = feedback_regression("Earth_Land")[0][simulation].mean(axis=1)
        whole_param = whole_df[param] * 12

        param = df.params()[simulation][param]
        param_mean = param.mean(axis=1) * 12
        param_std = param.std(axis=1) * 12

        ax[subplot] = fig.add_subplot(subplot)

        ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=0.8)
        ax[subplot].bar(param.index + 5 + bw / 2,
                        param_mean,
                        yerr=param_std * 1.645,
                        width=abs(bar_width),
                        color=color,
                        label=label)
        ax[subplot].axhline(y= whole_param, ls='--', color=color, alpha=0.8, lw=1)
        ax[subplot].legend()

    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel('Feedback parameter   (yr$^{-1}$)', fontsize=16, labelpad=20)

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + "fb_trendy_month.png")

def fb_regional_trendy(save=False):
    uptake = fb_id.trendy_duptake

    all_subplots = ['211', '212']
    # Note u_gamma is used to compare with beta.
    parameters = [['beta', 'S1'], ['u_gamma', 'S3']]

    vars = ['Earth', 'South', 'Tropical', 'North']
    colors = ['gray', 'blue', 'orange', 'green']
    bar_width = 1.25
    bar_position = np.arange(-2, 2) * bar_width

    fig = plt.figure(figsize=(16,6))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)


    for subplot, parameter in zip(all_subplots, parameters):
        for var, color, bar_pos in zip(vars, colors, bar_position):
            df = FeedbackAnalysis.TRENDY(
                                        fb_id.co2['month'],
                                        fb_id.temp['month'],
                                        uptake,
                                        var + '_Land'
                                        )

            param = df.params()[parameter[1]][parameter[0]]
            param_mean = param.mean(axis=1) * 12
            param_std = param.std(axis=1) * 12

            ax[subplot] = fig.add_subplot(subplot)

            ax[subplot].bar(param.index + 5 + bar_pos,
                            param_mean,
                            yerr=param_std*1.645,
                            width=bar_width,
                            color=color)

    ax["211"].set_xlabel(r'$\beta$', fontsize=14, labelpad=5)
    ax["211"].xaxis.set_label_position('top')
    ax["212"].set_xlabel(r'$u_{\gamma}$', fontsize=14, labelpad=5)
    ax["212"].xaxis.set_label_position('top')

    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r'Feedback parameter   (yr $^{-1}$)',
                   fontsize=16,
                   labelpad=20
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + "fb_trendy_regional_month.png")

def fb_regional_trendy2(save=False):
    uptake = fb_id.trendy_duptake

    all_subplots = ['41'+str(i) for i in range(1,5)]
    vars = ['Earth', 'South', 'Tropical', 'North']

    # Note u_gamma is used to compare with beta.
    parameters = ['beta', 'u_gamma']
    simulations = ['S1', 'S3']
    colors = ['green', 'red']
    bar_width = 1.5
    bar_position = np.arange(-1, 1) * bar_width

    fig = plt.figure(figsize=(16,14))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ymin, ymax = [], []
    for subplot, var in zip(all_subplots, vars):
        df = FeedbackAnalysis.TRENDY(
                                    fb_id.co2['month'],
                                    fb_id.temp['month'],
                                    uptake,
                                    var + '_Land'
                                    )
        for parameter, sim, color, bar_pos in zip(parameters, simulations, colors, bar_position):
            param = df.params()[sim][parameter]
            param_mean = param.mean(axis=1) * 12
            param_std = param.std(axis=1) * 12

            ax[subplot] = fig.add_subplot(subplot)

            ax[subplot].bar(param.index + 5 + bar_pos,
                            param_mean,
                            yerr=2*param_std,
                            width=bar_width,
                            color=color)

            if subplot != "411":
                ymin.append((param_mean - 1.645*param_std).min())
                ymax.append((param_mean + 1.645*param_std).max())

    delta = 0.05
    for subplot in ["412", "413", "414"]:
        ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                              max(ymax) + delta * abs(max(ymax))])

    for subplot, region in zip(["411", "412", "413", "414"],
                               ['Earth', 'South', 'Tropical', 'North']):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    axl.set_xlabel("Middle year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel(r'Feedback parameter   (yr$^{-1}$)',
                   fontsize=16,
                   labelpad=40
                  )

    if save:
        plt.savefig(fb_id.FIGURE_DIRECTORY + "fb_trendy_regional_month2.png")


""" EXECUTION """
whole_param = feedback_regression('Earth_Land')
whole_param[0]['S1'].mean(axis=1)*12
whole_param[0]['S3'].mean(axis=1)*12



whole_param[1]['S1']
whole_param[1]['S3']


medianreg = median_regstat("North_Land")
medianreg['S1']
medianreg['S3']


""" FIGURES """
fb_trendy(save=False)

fb_regional_trendy(save=False)

fb_regional_trendy2(save=False)
