""" Development of seasonal uptake using monthly data of inversion and
TRENDY uptake.
"""

from scipy import signal
import pandas as pd
import numpy as np
import xarray as xr
from copy import deepcopy

from core import inv_flux as invf
from core import trendy_flux as TRENDYf

from importlib import reload
reload(invf);

import os

FIGURE_DIRECTORY = "./../../../latex/thesis/figures/"
INV_DIRECTORY = "./../../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../../output/TRENDY/spatial/output_all/"
TRENDY_MEAN_DIRECTORY = "./../../../output/TRENDY/spatial/mean_all/"

year_invf = {}
month_invf = {}
for model in os.listdir(INV_DIRECTORY):
    model_dir = INV_DIRECTORY + model + '/'
    year_invf[model] = xr.open_dataset(model_dir + 'year.nc')
    month_invf[model] = xr.open_dataset(model_dir + 'month.nc')

year_S1_trendy = {}
year_S3_trendy = {}
month_S1_trendy = {}
month_S3_trendy = {}
for model in os.listdir(TRENDY_DIRECTORY):
    model_dir = TRENDY_DIRECTORY + model + '/'

    if 'S1' in model:
        year_S1_trendy[model] = xr.open_dataset(model_dir + 'year.nc')
        try:
            month_S1_trendy[model] = xr.open_dataset(model_dir + 'month.nc')
        except FileNotFoundError:
            pass
    elif 'S3' in model:
        year_S3_trendy[model] = xr.open_dataset(model_dir + 'year.nc')
        try:
            month_S3_trendy[model] = xr.open_dataset(model_dir + 'month.nc')
        except FileNotFoundError:
            pass


reload(invf);
reload(TRENDYf);


def seasonal_uptake(ds):
    """If data is monthly resolved, split the dataset into negative values
    (representing uptake in the winter season) and positive values
    (representing uptake in the summer season) for use of seasonal analysis.
    Note that seasons are calibrated to boreal geography because boreal
    seasons are more dominant than austral seasons.

    Returns
    -------

    summer: xr.Dataset

        Annual sum of uptake during the summer season.

    winter: xr.Dataset

        Annual sum of uptake during the winter season.

    """

    # if not hasattr(self, 'lat_split_ds'):
    #     self.lat_split_ds = self.latitudinal_splits()
    #
    # ds = self.lat_split_ds

    index = pd.to_datetime(ds.time.values).year.unique()
    variables = [
        'Earth_Land', 'South_Land', 'Tropical_Land', 'North_Land']

    summer, winter = {}, {}
    for variable in variables:
        winter[variable], summer[variable] = [], []
        for time in index:
            year = ds[variable].sel(time=str(time)).values
            winter[variable].append(year[year < 0].sum())
            summer[variable].append(year[year >= 0].sum())

    summer_ds = xr.Dataset(
        {key: (('time'), value) for (key, value) in summer.items()},
        coords={'time': (('time'), pd.to_datetime(index, format='%Y'))}
    )
    winter_ds = xr.Dataset(
        {key: (('time'), value) for (key, value) in winter.items()},
        coords={'time': (('time'), pd.to_datetime(index, format='%Y'))}
    )

    return {'summer': summer_ds, 'winter': winter_ds}

seasonal_uptake(ds)

dfinv = invf.SpatialAgg("./../../../data/inversions/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc")
dfinv.seasonal_uptake()


del df
df = TRENDYf.SpatialAgg('./../../../data/TRENDY/models/VISIT/S3/VISIT_S3_nbp.nc')

df.latitudinal_splits()
dsr = df.seasonal_uptake()

dsr['summer']

abs(dsr['winter'].Earth_Land).plot()
dsr['summer'].Earth_Land.plot()


ds = df.latitudinal_splits()

ds


time = pd.to_datetime(ds.time.values).year.unique()
summer = []
winter = []
for year in time:
    year_val = ds.Earth_Land.sel(time=str(year)).values
    winter.append(year_val[year_val<0].sum())
    summer.append(year_val[year_val>=0].sum())

month_invf['CAMS'].Earth_Land.resample({'time': 'Y'}).sum().values

pd.DataFrame({'summer': summer, 'winter': winter}, index=time).sum(axis=1).values


pd.DataFrame({'summer': summer, 'winter': winter}, index=time)
