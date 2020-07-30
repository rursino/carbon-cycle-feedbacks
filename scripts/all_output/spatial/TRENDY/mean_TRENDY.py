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


""" INPUTS """
PATH_DIR = './../../../../output/TRENDY/spatial/output_all/'
OUTPUT_DIR = './../../../../output/TRENDY/spatial/mean_all/'

sims = ['S1', 'S3']
timeres = ['month', 'year', 'decade', 'whole']


""" FUNCTIONS """
def open_dataframes(sim, timeres):
    """ Generates the appropriate filenames from chosen simulation (sim) and
    time resolution (timeres) and opens and stores all the datasets for each
    filenames in a list.
    """

    models = ['CABLE-POP', 'JSBACH', 'CLASS-CTEM', 'LPJ-GUESS', 'OCN']
    fnames = [PATH_DIR + f'{model}_{sim}_nbp/{timeres}.nc' for model in models]

    # LPJ-GUESS has no month dataset.
    if timeres == 'month':
        fnames.remove(PATH_DIR + f'LPJ-GUESS_{sim}_nbp/month.nc')

    return [xr.open_dataset(fname) for fname in fnames]

def generate_mean_dataset(dataframes):
    """ Generates a dataset that takes the mean of all models for each
    simulation and time resolution (dataframes taken in as input 'dataframes').
    """

    variables = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']

    values = {}
    for variable in variables:
        vals = 0
        for ds in dataframes:
            vals += ds.sel(time=slice("1701", "2018"))[variable].values

        values[variable] = vals / len(dataframes)

    time = dataframes[0].sel(time=slice("1701", "2018")).time.values

    ds = xr.Dataset(
        {key: (('time'), value) for (key, value) in values.items()},
        coords={'time': (('time'), time)}
    )

    return ds


""" EXECUTION """
# Generate mean dataset for each simulation and timeresolution and save as
# netCDF files in spatial/mean_all directory.
for sim, time in tqdm(product(sims, timeres)):
    dataframes = open_dataframes(sims[0], timeres[0])
    ds = generate_mean_dataset(dataframes)

    try:
        ds.to_netcdf(OUTPUT_DIR + f'{sim}/{time}.nc')
    except:
        os.mkdir(OUTPUT_DIR + f'{sim}/')
        ds.to_netcdf(OUTPUT_DIR + f'{sim}/{time}.nc')
