""" Ensure that regional contributions of feedback parameters add up to the
global contribution.
"""

from core import FeedbackAnalysis
import numpy as np
import pandas as pd
import xarray as xr
import os
from itertools import *

from importlib import reload
reload(FeedbackAnalysis)

DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'
INV_DIRECTORY = "./../../../output/inversions/spatial/output_all/"


co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}


temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}


invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}, 'month': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')


variables = ['Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']
fa = {}
for var in variables:
    fa[var] = FeedbackAnalysis.INVF(co2['year'], temp['year'], invf_uptake['year'], var).params()['beta']


def check_sum(model, year):
     earth = fa['Earth_Land'][model][year]
     south = fa['South_Land'][model][year]
     tropical = fa['Tropical_Land'][model][year]
     north = fa['North_Land'][model][year]

     return earth - south - tropical - north

for m, y in product(['CAMS', 'Rayner', 'JENA_s76'], [1980, 1990, 2000]):
    print(check_sum(m,y))
