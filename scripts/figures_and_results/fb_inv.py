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
                                "T": T[variable.split('_')[0]]
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
def fb_inv(timeres):
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

        return beta, gamma

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

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel('Feedback parameter   (yr$^{-1}$)', fontsize=16, labelpad=40)


""" EXECUTION """
reg_models, model_stats = feedback_regression('month', 'North_Land')
reg_models
model_stats.mean()

all_regstat("month", "Earth_Land")['CAMS']
mean_regstat("year", "Earth_Land")
mean_regstat("month", "Earth_Land")

fb_inv('year')
