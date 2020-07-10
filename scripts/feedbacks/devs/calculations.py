""" Calculation of beta and gamma with methodology via methods.txt.
"""

""" IMPORTS """

import xarray as xr
import pandas as pd
import numpy as np
import statsmodels as sm


""" INPUTS """
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'

# Uptake - inversions
U = xr.open_dataset(OUTPUT_DIR + 'inversions/spatial/output_all/CAMS/year.nc')
C = pd.read_csv(DIR + 'data/CO2/co2_global.csv', index_col='Year')['CO2']
T = xr.open_dataset(OUTPUT_DIR + 'TEMP/spatial/output_all/CRUTEM/year.nc')


U_time = pd.to_datetime([time.year for time in U.time.values], format='%Y').year
C_time = C.index
T_time = pd.to_datetime(T.time.values).year

def time_intersection(*args):
    """ Grabs a set of pd.DatetimeIndex objects and returns the intersection of
    time points. This makes indexing dataframes and datasets to match time
    points accessible.

    Parameters
    ==========

    *args: pd.DateTimeIndex

        A set of arguments to pass pd.DateTimeIndex objects for matching to
        other objects.

    """

    intersection_index = args[0]
    length_of_args = len(args)
    i = 1
    while i < length_of_args:
         intersection_index = intersection_index.intersection(args[i])
         i += 1

    return intersection_index

intersection_index = time_intersection(U_time, C_time, T_time)
