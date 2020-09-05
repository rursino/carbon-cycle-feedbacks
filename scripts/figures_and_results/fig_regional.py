""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
from scipy import signal
import pandas as pd
import numpy as np
import xarray as xr
from copy import deepcopy

import seaborn as sns
sns.set_style('darkgrid')

import sys
from core import inv_flux as invf
from core import trendy_flux as TRENDYf

import os


""" FUNCTIONS """
def deseasonalise_instance(instance_dict):
    d_instance_dict = deepcopy(instance_dict)
    for model in d_instance_dict:
        d_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].deseasonalise(key)) for
            key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',
                    'Earth_Ocean', 'South_Ocean', 'North_Ocean', 'Tropical_Ocean']},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return d_instance_dict

def bandpass_instance(instance_dict, fc):
    bp_instance_dict = deepcopy(instance_dict)

    for model in bp_instance_dict:
        bp_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].bandpass(key, fc)) for
            key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',
                    'Earth_Ocean', 'South_Ocean', 'North_Ocean', 'Tropical_Ocean']},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return bp_instance_dict

def bandpass_filter(x, btype, fc):
    fs = 12
    order = 5

    if btype == "band":
        assert type(fc) == list, "fc must be a list of two values."
        fc = np.array(fc)

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(order, w, btype)

    return signal.filtfilt(b, a, x)


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"
TRENDY_MEAN_DIRECTORY = "./../../output/TRENDY/spatial/mean_all/"

year_invf = {}
month_invf = {}
for model in os.listdir(INV_DIRECTORY):
    model_dir = INV_DIRECTORY + model + '/'
    year_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
    month_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'month.nc'))


dmonth_invf = deseasonalise_instance(month_invf)
bpmonth_invf = bandpass_instance(dmonth_invf, 1/(25*12))

year_S1_trendy = {}
year_S3_trendy = {}
month_S1_trendy = {}
month_S3_trendy = {}
for model in os.listdir(TRENDY_DIRECTORY):
    model_dir = TRENDY_DIRECTORY + model + '/'

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
    "S1": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S1/year.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S3/year.nc'))
}
month_mean = {
    "S1": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S1/month.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S3/month.nc'))
}

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]


""" FIGURES """
def inv_regional_cwt(timeres="year", save=False):
    invf_dict = {
        "year": year_invf,
        "month": month_invf
    }

    variables = [['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land'],
                 ['Earth_Ocean', 'South_Ocean', 'Tropical_Ocean', 'North_Ocean']]
    all_subplots = [['42' + str(num) for num in range(i,i+8,2)] for i in range(1,3)]
    colors = ['black', 'blue', 'orange', 'green']

    zip_list = zip(all_subplots, variables)

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplots, vars in zip_list:
        ymin, ymax = [], []
        for subplot, var, color in zip(subplots, vars, colors):
            ax[subplot] = fig.add_subplot(subplot)

            index, vals = [], []
            for model in invf_dict[timeres]:
                df = invf_dict[timeres][model].cascading_window_trend(variable = var, indep="CO2", window_size=10)
                index.append(df.index)
                vals.append(df.values.squeeze())

            pad_max_length = 12 if timeres == "month" else 6
            dataframe = {}
            for ind, val in zip(index, vals):
                if timeres == "month":
                    val = bandpass_filter(val, 'low', 365/667)
                for i, v in zip(ind, val):
                    if i in dataframe.keys():
                        dataframe[i].append(v)
                    else:
                        dataframe[i] = [v]
            for ind in dataframe:
                row = np.array(dataframe[ind])
                dataframe[ind] = np.pad(row, (0, 12 - len(row)), 'constant', constant_values=np.nan)

            df = pd.DataFrame(dataframe).T.sort_index().iloc[3:]
            df *= 1e3  # if want MtC yr-1 uptake

            x = df.index
            y = df.mean(axis=1)
            std = df.std(axis=1)

            ax[subplot].plot(x, y, color=color)
            ax[subplot].fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)
            ax[subplot].axhline(ls='--', color='k', alpha=0.5, lw=1)

            stat_vals[var] = stats.linregress(x, y)

            ymin.append((y - 2*std).min())
            ymax.append((y + 2*std).max())

        delta = 0.25
        for subplot in subplots:
            ax[subplot].set_ylim([min(ymin) - delta * abs(min(ymin)),
                                  max(ymax) + delta * abs(max(ymax))])

    ax["421"].set_xlabel("Land", fontsize=14, labelpad=5)
    ax["421"].xaxis.set_label_position('top')
    ax["422"].set_xlabel("Ocean", fontsize=14, labelpad=5)
    ax["422"].xaxis.set_label_position('top')
    for subplot, region in zip(["421", "423", "425", "427"],
                               ["Earth", "South", "Tropical", "North"]):
        ax[subplot].set_ylabel(region, fontsize=14, labelpad=5)

    axl.set_xlabel("First value of CO$_2$ window (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$  " + " (MtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=35)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"inv_regional_{timeres}_cwt.png")

    return stat_vals


inv_regional_cwt("month")





def trendy_regional_year_cwt(save=False):
    vars = ['South_Land', 'Tropical_Land', 'North_Land']
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
        for var, color in zip(vars, ['blue', 'orange', 'green']):
            index, vals = [], []
            for model in trendy_year_sim:
                df = trendy_year_sim[model].cascading_window_trend(
                                                variable = var,
                                                indep="CO2",
                                                window_size=10)
                index.append(df.index)
                vals.append(df.values.squeeze())

            if timeres == "month":
                dataframe = {}
                    for i, v in zip(df.index, df.values.squeeze()):
                        filterv = bandpass_filter(v, 'low', 365/667)
                        if i in dataframe.keys():
                            dataframe[i].append(filterv)
                        else:
                            dataframe[i] = [filterv]
                    for ind in dataframe:
                        row = np.array(dataframe[ind])
                        dataframe[ind] = np.pad(row, (0, 12 - len(row)), 'constant', constant_values=np.nan)

            else:
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

            ax.plot(x, y, color=color)
            # ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

            regstats = stats.linregress(x, y)
            # slope, intercept, rvalue, pvalue, _ = regstats
            # ax.plot(x, x*slope + intercept)
            # text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
            # xtext = x.min() + 0.75 * (x.max() -x.min())
            # ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
            # ax.text(xtext, ytext, text, fontsize=15)

            stat_vals.append(regstats)

        plt.legend(vars)
        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

    axl.set_title(f"Cascading Window 10-Year Trend - TRENDY",
                 fontsize=30, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy_year_cwt.png")

    return stat_vals

def trendy_regional_month_cwt(save=False):
    vars = ['South_Land', 'Tropical_Land', 'North_Land']
    simulations = ['S1', 'S3']

    fig = plt.figure(figsize=(16,12))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    subplots = ['211', '212']
    stat_vals = []
    for sim, subplot in zip(simulations, subplots):
        trendy_year_sim = month_mean[sim]

        ax = fig.add_subplot(subplot)
        for var, color in zip(vars, ['blue', 'orange', 'green']):
            df = trendy_year_sim.cascading_window_trend(
                                            variable = var,
                                            indep="CO2",
                                            window_size=10)

            dataframe = {}
            for i, v in zip(df.index, df.values.squeeze()):
                if i in dataframe.keys():
                    dataframe[i].append(v)
                else:
                    dataframe[i] = [v]
            for ind in dataframe:
                row = np.array(dataframe[ind])
                dataframe[ind] = np.pad(row, (0, 12 - len(row)), 'constant', constant_values=np.nan)

            df = pd.DataFrame(dataframe).T.sort_index()
            x = df.index
            y = df.mean(axis=1)
            std = df.std(axis=1)

            ax.plot(x, y, color=color)
            # ax.fill_between(x, y - 2*std, y + 2*std, color='gray', alpha=0.2)

            regstats = stats.linregress(x, y)
            # slope, intercept, rvalue, pvalue, _ = regstats
            # ax.plot(x, x*slope + intercept)
            # text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
            # xtext = x.min() + 0.75 * (x.max() -x.min())
            # ytext = (y-2*std).min() +  0.8 * ((y+2*std).max() - (y-2*std).min())
            # ax.text(xtext, ytext, text, fontsize=15)

            stat_vals.append(regstats)

        plt.legend(vars)
        ax.axhline(ls='--', color='k', alpha=0.5, lw=1)

    axl.set_title(f"Cascading Window 10-Year Trend - TRENDY",
                 fontsize=30, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + f"trendy_year_cwt.png")

    return stat_vals


""" EXECUTION """
inv_regional_cwt("month", save=False)

inv_regional_month_cwt(bpmonth_invf, save=False)

trendy_regional_year_cwt(save=False)

trendy_regional_month_cwt(save=False)
