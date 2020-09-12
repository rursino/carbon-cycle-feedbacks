""" This script takes each dataset (corresponding to time resolution) for all
models and outputs a single dataset of the mean and standard deviation fluxes.
The simulations S1 and S3 are separated.
"""


""" IMPORTS """
import numpy as np
import xarray as xr
import pandas as pd
from itertools import *
from tqdm import tqdm
import os

import matplotlib.pyplot as plt


""" INPUTS """
CURRENT_DIR = os.path.dirname(__file__)

PATH_DIR = CURRENT_DIR + './../../../../output/TRENDY/spatial/output_all/'
OUTPUT_DIR = CURRENT_DIR + './../../../../output/TRENDY/spatial/mean_all/'

sims = ['S1', 'S3']
timeres = ['month', 'year']


""" FUNCTIONS """
def open_dataframes(sim, timeres):
    """ Generates the appropriate filenames from chosen simulation (sim) and
    time resolution (timeres) and opens and stores all the datasets for each
    filenames in a list.
    """

    models = list(set([model.split('_')[0] for model in os.listdir(PATH_DIR)]))
    fnames = {model:PATH_DIR + f'{model}_{sim}_nbp/{timeres}.nc' for model in models}

    dataframes = {model : xr.open_dataset(fnames[model]).sel(time=slice("1701", "2017")) for
            model in fnames if model != 'VISIT'}

    dataframes['VISIT'] = xr.open_dataset(fnames['VISIT']).sel(time=slice("1860", "2017"))

    return dataframes

def generate_mean_dataset(dataframes):
    """ Generates a dataset that takes the mean of all models for each
    simulation and time resolution (dataframes taken in as input 'dataframes').
    """

    variables = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']

    values = {}
    for variable in variables:
        vals = []
        for model in dataframes:
            ds = dataframes[model]
            if model == 'VISIT':
                VISIT_val = ds[variable].values
                val = np.pad(VISIT_val, (len(vals[0]) - len(VISIT_val), 0),
                             'constant', constant_values=np.nan)
                vals.append(val)
            else:
                vals.append(ds[variable].values)


        stack = np.stack(vals)
        mean = np.nanmean(stack, axis=0)
        std = np.nanstd(stack, axis=0)

        values[variable] = mean
        values[variable + '_STD'] = std

    time = list(dataframes.values())[0].time.values

    ds = xr.Dataset(
        {key: (('time'), value) for (key, value) in values.items()},
        coords={'time': (('time'), time)}
    )

    return ds


""" EXECUTION """
# Generate mean dataset for each simulation and timeresolution and save as
# netCDF files in spatial/mean_all directory.
for sim, time in tqdm(product(sims, timeres)):
    dataframes = open_dataframes(sim, time)
    ds = generate_mean_dataset(dataframes)

    try:
        ds.to_netcdf(OUTPUT_DIR + f'{sim}/{time}.nc')
    except:
        os.mkdir(OUTPUT_DIR + f'{sim}/')
        ds.to_netcdf(OUTPUT_DIR + f'{sim}/{time}.nc')
