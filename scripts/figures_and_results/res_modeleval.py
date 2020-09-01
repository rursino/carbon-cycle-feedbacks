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

import os


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/inversions/spatial/output_all/"

inv_modeleval = {}
for model in os.listdir(SPATIAL_DIRECTORY):
    model_dir = SPATIAL_DIRECTORY + model + '/'
    inv_modeleval[model] = invf.ModelEvaluation(xr.open_dataset(model_dir + 'year.nc'))


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


""" EXECUTION """
inv_regress_timeseries()["ocean"]

inv_regress_CWT(10)["land"]

inv_compare_trend()['ocean']
