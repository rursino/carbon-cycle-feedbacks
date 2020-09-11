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
from core import inv_flux as invf
from core import trendy_flux as TRENDYf

import os


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"

inv_modeleval = {}
for model in os.listdir(SPATIAL_DIRECTORY):
    model_dir = SPATIAL_DIRECTORY + model + '/'
    inv_modeleval[model] = invf.ModelEvaluation(xr.open_dataset(model_dir + 'year.nc'))

trendy_modeleval = {}
for model in os.listdir(TRENDY_DIRECTORY):
    model_dir = TRENDY_DIRECTORY + model + '/'
    trendy_modeleval[model] = TRENDYf.ModelEvaluation(xr.open_dataset(model_dir + 'year.nc'))


""" FUNCTIONS """
def inv_regress_timeseries():
    land_stats = {}
    ocean_stats = {}
    for model in inv_modeleval:
        land_stats[model] = inv_modeleval[model].regress_timeseries_to_GCP("land")
        ocean_stats[model] = inv_modeleval[model].regress_timeseries_to_GCP("ocean")

    stats = {}
    for var, stat_dict in zip(("land", "ocean"), (land_stats, ocean_stats)):
        df = pd.DataFrame(stat_dict).iloc[:4].T
        df.columns = ['slope', 'intercept', 'rvalue', 'pvalue']
        stats[var] = df

    return stats

def inv_regress_CWT(window_size):
    land_stats = {}
    ocean_stats = {}
    for model in inv_modeleval:
        land_stats[model] = inv_modeleval[model].regress_cascading_window_trend_to_GCP("land", window_size, indep="CO2")
        ocean_stats[model] = inv_modeleval[model].regress_cascading_window_trend_to_GCP("ocean", window_size, indep="CO2")

    stats = {}
    for var, stat_dict in zip(("land", "ocean"), (land_stats, ocean_stats)):
        df = pd.DataFrame(stat_dict).iloc[:4].T
        df.columns = ['slope', 'intercept', 'rvalue', 'pvalue']
        stats[var] = df

    return stats

def inv_compare_trend():
    land_stats = {}
    ocean_stats = {}
    for model in inv_modeleval:
        land_stats[model] = inv_modeleval[model].compare_trend_to_GCP("land")
        ocean_stats[model] = inv_modeleval[model].compare_trend_to_GCP("ocean")

    stats = {}
    for var, stat_dict in zip(("land", "ocean"), (land_stats, ocean_stats)):
        df = pd.DataFrame(stat_dict).T
        df.columns = ['GCP_slope', 'model_slope', 'diff']
        stats[var] = df

    return stats

def mean_inv_trend(compare_stats):
    stats = {}
    for region in ['land', 'ocean']:
        stats[region] = compare_stats[region][['GCP_slope', 'model_slope']].mean()
        stats[region]['diff'] = (stats[region]['model_slope'] - stats[region]['GCP_slope'])  * 100 / abs(stats[region]['GCP_slope'])

    return stats

def std_inv_trend(compare_stats):
    stats = {}
    for region in ['land', 'ocean']:
        stats[region] = compare_stats[region][['GCP_slope', 'model_slope']].std()
        stats[region]['diff'] = (stats[region]['model_slope'] - stats[region]['GCP_slope'])  * 100 / abs(stats[region]['GCP_slope'])

    return stats

def trendy_regress_timeseries():
    stats = {}
    for model in trendy_modeleval:
        stats[model] = trendy_modeleval[model].regress_timeseries_to_GCP()

    df = pd.DataFrame(stats).iloc[:4].T
    df.columns = ['slope', 'intercept', 'rvalue', 'pvalue']

    return df

def trendy_regress_CWT(window_size):
    stats = {}
    for model in trendy_modeleval:
        stats[model] = trendy_modeleval[model].regress_cascading_window_trend_to_GCP(window_size=window_size, indep="CO2")

    df = pd.DataFrame(stats).iloc[:4].T
    df.columns = ['slope', 'intercept', 'rvalue', 'pvalue']

    return df

def trendy_compare_trend():
    stats = {}
    for model in trendy_modeleval:
        stats[model] = trendy_modeleval[model].compare_trend_to_GCP()

    df = pd.DataFrame(stats).T
    df.columns = ['GCP_slope', 'model_slope', 'diff']

    return df

def mean_trendy_trend(compare_stats):

    df = compare_stats[['GCP_slope', 'model_slope']].mean()
    df['diff'] = (df['model_slope'] - df['GCP_slope'])  * 100 / abs(df['GCP_slope'])

    return df

def std_trendy_trend(compare_stats):

    df = compare_stats[['GCP_slope', 'model_slope']].std()
    df['diff'] = (df['model_slope'] - df['GCP_slope'])  * 100 / abs(df['GCP_slope'])

    return df


""" INVF EXECUTION """
inv_regress_timeseries()["ocean"]
inv_regress_timeseries()["ocean"].mean()
inv_regress_timeseries()["ocean"].std()

inv_regress_CWT(10)["land"]

inv_trend = inv_compare_trend()
inv_trend['land']
mean_inv_trend(inv_trend)['ocean']
std_inv_trend(inv_trend)['ocean']


""" TRENDY EXECUTION """
trendy_regress_timeseries()
# trendy_regress_timeseries().mean()
# trendy_regress_timeseries().std()

trendy_regress_CWT(10)

trendy_trend = trendy_compare_trend()
trendy_trend
mean_trendy_trend(trendy_trend)
std_trendy_trend(trendy_trend)
