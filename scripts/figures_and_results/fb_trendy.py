""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels import api as sm
from scipy import signal

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import os
from core import FeedbackAnalysis


""" INPUTS """
DIR = './../../'
OUTPUT_DIR = DIR + 'output/'

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

model_names = ['LPJ-GUESS', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in model_names},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in model_names if model_name != "LPJ-GUESS"}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in model_names},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in model_names if model_name != "LPJ-GUESS"}
    }
}

phi, rho = 0.015, 1.93


""" FUNCTIONS """
def deseasonalise(x):
    """
    """

    fs = 12
    fc = 365/667

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'low')

    return signal.filtfilt(b, a, x)

def feedback_regression(timeres, variable):
    start, end = 1960, 2017
    reg_models = {}
    model_stats = {}
    for simulation in ['S1', 'S3']:
        if timeres == "month":
            duptake = uptake[simulation][timeres]
            for model in duptake:
                uptake[simulation][timeres][model] = xr.Dataset(
                    {key: (('time'), deseasonalise(duptake[model][key].values)) for
                    key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']},
                    coords={'time': (('time'), duptake[model].time.values)}
                )

        model_names = uptake[simulation][timeres].keys()

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
            C = co2[timeres].loc[start:end]
            T = temp[timeres].sel(time=slice(str(start), str(end)))
            U = uptake[simulation][timeres][model_name].sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T[variable.split('_')[0]]
                                    }
                                   )

            X = sm.add_constant(df[["C", "T"]])
            Y = df["U"]

            reg_model = sm.OLS(Y, X).fit()

            sim_reg_models[model_name] = reg_model.params[['C', 'T']].values
            sim_model_stats['r_squared'].append(reg_model.rsquared)
            sim_model_stats['t_values_beta'].append(reg_model.tvalues.loc['C'])
            sim_model_stats['t_values_gamma'].append(reg_model.tvalues.loc['T'])
            sim_model_stats['p_values_beta'].append(reg_model.pvalues.loc['C'])
            sim_model_stats['p_values_gamma'].append(reg_model.pvalues.loc['T'])
            sim_model_stats['mse_total'].append(reg_model.mse_total)
            sim_model_stats['nobs'].append(reg_model.nobs)

        reg_models[simulation] = pd.DataFrame(sim_reg_models, index=['beta', 'gamma'])

        reg_models[simulation].loc['beta'] /= 2.12
        reg_models[simulation].loc['gamma'] *= phi / rho

        model_stats[simulation] = pd.DataFrame(sim_model_stats, index=model_names)

    return reg_models, model_stats

def all_regstat(timeres, variable):
    all_regstats = {}
    for simulation in ["S1", "S3"]:
        df = FeedbackAnalysis.TRENDY(
                                    co2[timeres],
                                    temp[timeres],
                                    uptake[simulation][timeres],
                                    variable
                                    )
        all_regstats[simulation] = {model : pd.DataFrame(df.regstats()[model],
                                        index=df.regstats()['Year'])
                   for model in model_names
               }

    return all_regstats

def mean_regstat(timeres, variable):
    mean_regstats = {}
    for simulation in ["S1", "S3"]:
        df = FeedbackAnalysis.TRENDY(
                                    co2[timeres],
                                    temp[timeres],
                                    uptake[simulation][timeres],
                                    variable
                                    )
        regstat = [pd.DataFrame(df.regstats()[model],
                                index=df.regstats()['Year'])
                   for model in model_names if model != 'LPJ-GUESS'
                  ]

        mean_regstats[simulation] = regstat[0]
        for i in range(1, len(regstat)):
            mean_regstats[simulation] += regstat[i]

        mean_regstats[simulation] /= len(regstat)

    return mean_regstats


""" FIGURES """
def fb_trendy(timeres):
    fig = plt.figure(figsize=(14,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ylim = []
    for subplot, simulation in zip(["211", "212"], ["S1", "S3"]):
        df = FeedbackAnalysis.TRENDY(
                                    co2[timeres],
                                    temp[timeres],
                                    uptake[simulation][timeres],
                                    "Earth_Land"
                                    )
        beta, gamma = df.params()

        beta_mean = beta.mean(axis=1)
        beta_std = beta.std(axis=1)
        gamma_mean = gamma.mean(axis=1)
        gamma_std = gamma.std(axis=1)

        ylim.append([min(beta_mean.min(), gamma_mean.min()),
                     max(beta_mean.max(), gamma_mean.max())
                    ])

        ax[subplot] = fig.add_subplot(subplot)

        bar_width = 3
        if simulation == "S3" or "S1":
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
        # elif simulation == "S1":
        #     ax[subplot].bar(beta.index + 5,
        #                     beta_mean,
        #                     yerr=beta_std,
        #                     width=bar_width,
        #                     color='green')

        ax[subplot].legend([r'$\beta$', r'$u_{\gamma}$'])
        ax[subplot].set_ylabel(simulation, fontsize=14, labelpad=5)

    axl.set_xlabel("First year of 10-year window", fontsize=16, labelpad=10)
    axl.set_ylabel('Feedback parameter   (yr$^{-1}$)', fontsize=16, labelpad=40)


""" EXECUTION """
reg_models, model_stats = feedback_regression('month', 'North_Land')
reg_models['S3']
model_stats['S3'].mean(axis=0)

all_regstat("year", "Earth_Land")["S3"]["OCN"]
mean_regstat("year", "Earth_Land")

fb_trendy('month')
