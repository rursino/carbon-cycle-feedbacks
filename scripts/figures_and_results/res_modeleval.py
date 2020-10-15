""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
import xarray as xr
from copy import deepcopy
from itertools import *

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

GCP = pd.read_csv('./../../data/GCP/budget.csv', index_col='Year')[['ocean sink', 'land sink']]

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

def model_bias(model_set, models, sink):
    """ Calculate model bias based on formula above.

    Parameters:
    -----------

    model_set: dict

        Either invf_modeleval or trendy_modeleval

    models: list-like

        List of two models to use.

    sink: str

        Either land or ocean

    """

    models_df = {}

    for model in models:
        model_df = model_set[model].data
        start = pd.to_datetime(model_df.time.values[0]).year
        end = pd.to_datetime(model_df.time.values[-1]).year

        gcp_df = GCP.loc[start:end]

        models_df[model] = {'model': model_df[f'Earth_{sink.title()}'].mean(),
                            'GCP': gcp_df[f'{sink} sink'].mean()}

    return abs((models_df[models[1]]['model'] - models_df[models[0]]['model'])
                - (models_df[models[1]]['GCP'] - models_df[models[0]]['GCP'])).values


""" INVF EXECUTION """
inv_regress_timeseries()["ocean"]
inv_regress_timeseries()["ocean"].mean()
inv_regress_timeseries()["ocean"].std()

# inv_regress_CWT(10)["land"]

inv_trend = inv_compare_trend()
inv_trend['ocean']
mean_inv_trend(inv_trend)['ocean']
inv_trend['land'].std()


{models:model_bias(inv_modeleval, models, 'land') for models in combinations(inv_modeleval.keys(), 2)}
{models:model_bias(inv_modeleval, models, 'ocean') for models in combinations(inv_modeleval.keys(), 2)}



""" TRENDY EXECUTION """
s3_timeseries = trendy_regress_timeseries().loc[('S3' in i for i in trendy_regress_timeseries().index)]
s3_timeseries
s3_timeseries.mean()
s3_timeseries.std()

# trendy_regress_CWT(10)

trendy_trend = trendy_compare_trend().loc[('S3' in i for i in trendy_compare_trend().index)]
trendy_trend
mean_trendy_trend(trendy_trend)
trendy_trend.std()


trendy_key = ['VISIT_S3_nbp', 'JSBACH_S3_nbp', 'CABLE-POP_S3_nbp',
              'OCN_S3_nbp', 'CLASS-CTEM_S3_nbp']
{models:model_bias(trendy_modeleval, models, 'land') for models in combinations(trendy_key, 2)}
