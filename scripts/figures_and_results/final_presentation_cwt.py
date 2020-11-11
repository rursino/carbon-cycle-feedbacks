""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import numpy as np
import xarray as xr

import seaborn as sns
sns.set_style('darkgrid')

import sys
from copy import deepcopy
sys.path.append('../core')
import GCP_flux as GCPf
import inv_flux as invf
import trendy_flux as TRENDYf

import os

from importlib import reload
reload(GCPf);
reload(invf);
reload(TRENDYf);


""" INPUTS """
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"

year_invf = {}
for model in os.listdir(INV_DIRECTORY):
    model_dir = INV_DIRECTORY + model + '/'
    year_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'year.nc'))

year_S1_trendy = {}
year_S3_trendy = {}
for model in os.listdir(TRENDY_DIRECTORY):
    model_dir = TRENDY_DIRECTORY + model + '/'

    if 'S1' in model:
        year_S1_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))

    elif 'S3' in model:
        year_S3_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]

land_GCPf = GCPf.Analysis("land sink")
ocean_GCPf = GCPf.Analysis("ocean sink")
land = land_GCPf.data
ocean = ocean_GCPf.data

df = land_GCPf.all_data


""" CWT """
def inv_year_cwt():
    sink = {}
    for var in ['Earth_Land', 'Earth_Ocean']:
        index, vals = [], []
        for model in year_invf:
            df = year_invf[model].cascading_window_trend(variable = var,
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

        sink[var] = df.mean(axis=1)

    return sink

def trendy_year_cwt():
    trendy_year = {
        'S1': year_S1_trendy,
        'S3': year_S3_trendy
    }

    regstats = {}
    trendy_cwt = {}
    for sim in trendy_year:
        trendy_year_sim = trendy_year[sim]

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

        trendy_cwt[sim] = df.mean(axis=1)

    return trendy_cwt

def cwt():
    inv_cwt = inv_year_cwt()
    trendy_cwt = trendy_year_cwt()

    gcp_list = zip(['311', '312'],
                   ['land', 'ocean'],
                   [land_GCPf, ocean_GCPf]
                   )

    inv_list = zip(['311', '312'],
                   ['land', 'ocean'],
                   [inv_cwt['Earth_Land'], inv_cwt['Earth_Ocean']])

    fig = plt.figure(figsize=(18,20))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    ax = {}
    regstats = {'GCP': {}, 'inv': {}}

    for subplot, var, sink in gcp_list:
        ax[subplot] = fig.add_subplot(subplot)

        sink_cwt_df = sink.cascading_window_trend(window_size=10)
        sink_cwt_df *= 1e3 # units MtC / yr / GtC
        x = sink_cwt_df.index
        y = sink_cwt_df.values

        ax[subplot].plot(x, y, color='black')
        ax[subplot].set_xlabel(f"{var.title()}", fontsize=24,
                        labelpad=5)
        ax[subplot].xaxis.set_label_position('top')

        stat = stats.linregress(x, y)
        regstats['GCP'][var] = (stat.slope, stat.rvalue, stat.pvalue)

    for subplot, var, sink in inv_list:
        x = sink.index
        y = sink.values
        ax[subplot].plot(x, y, color='brown')

        stat = stats.linregress(x, y)
        regstats['inv'][var] = (stat.slope, stat.rvalue, stat.pvalue)

    for sim, color in zip(trendy_cwt, ['blue', 'red']):
        trendy_subplot = '313'
        ax[trendy_subplot] = fig.add_subplot(trendy_subplot)
        df = trendy_cwt[sim]
        x = df.index
        y = df.values
        ax[trendy_subplot].plot(x, y, color=color)
        ax[trendy_subplot].set_xlabel("TRENDY", fontsize=32,
                            labelpad=5)
        ax[trendy_subplot].xaxis.set_label_position('top')

    axl.set_xlabel("First year of 10-year window", fontsize=36, labelpad=40)
    axl.set_ylabel(r"$\alpha$   " + " (MtC.yr$^{-1}$.GtC$^{-1}$)", fontsize=36,
                    labelpad=40)

    for subplot in ax:
        ax[subplot].tick_params(axis="x", labelsize=18)
        ax[subplot].tick_params(axis="y", labelsize=18)

    for subplot in ['311', '312']:
        ax[subplot].legend(['GCP', 'Inversions'], fontsize=20)
    ax['313'].legend(['S1', 'S3'], fontsize=20)

    fig.tight_layout(pad=5.0)

    return regstats


cwt()
