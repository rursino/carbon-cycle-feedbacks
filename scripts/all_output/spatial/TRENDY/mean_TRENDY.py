""" This script takes each dataset (corresponding to time resolution) for all
models and outputs a single dataset of the mean and standard deviation fluxes.
The simulations S1 and S3 are separated.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
from itertools import *

""" INPUTS """
PATH_DIRECTORY = './../../../../output/TRENDY/spatial/output_all/'

models = ['CABLE-POP', 'JSBACH', 'CLASS-CTEM', 'LPJ-GUESS', 'OCN']
sims = ['S1', 'S3']
timeres = ['month', 'year', 'decade', 'whole']

fname1 = PATH_DIRECTORY + f'{models[0]}_{sims[0]}_nbp/{timeres[0]}.nc'
df1 = xr.open_dataset(fname1)

fname2 = PATH_DIRECTORY + f'{models[2]}_{sims[0]}_nbp/{timeres[0]}.nc'
df2 = xr.open_dataset(fname2)

variables = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']

earth_vals = (df1[variables[0]].values + df2[variables[0]].values) / 2
df2.time.values




time = df1.time.values
values = {'Earth_Land': earth_vals}

ds = xr.Dataset(
    {key: (('time'), value) for (key, value) in values.items()},
    coords={'time': (('time'), time)}
)

ds
