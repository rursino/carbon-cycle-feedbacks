""" Store all functionality to import and pre-process relevant input data for
past_trends figures here."""

""" IMPORTS """
from scipy import signal
import pandas as pd
import numpy as np
import xarray as xr
from copy import deepcopy

from core import inv_flux as invf
from core import trendy_flux as TRENDYf

import os



""" FUNCTIONS """
def deseasonalise(x):
    """
    """

    fs = 12
    fc = 365/667

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'low')

    return signal.filtfilt(b, a, x)

def deseasonalise_instance(instance_dict):
    if 'JSBACH_S1_nbp' in instance_dict or 'JSBACH_S3_nbp' in instance_dict:
        vars = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']
    else:
        vars = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',
                'Earth_Ocean', 'South_Ocean', 'North_Ocean', 'Tropical_Ocean']
    d_instance_dict = deepcopy(instance_dict)
    for model in d_instance_dict:
        d_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].deseasonalise(key)) for
            key in vars},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return d_instance_dict

def bandpass_filter(x, btype, fc):
    fs = 12
    order = 5

    if btype == "band":
        assert type(fc) == list, "fc must be a list of two values."
        fc = np.array(fc)

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(order, w, btype)

    return signal.filtfilt(b, a, x)

def bandpass_instance(instance_dict, fc, btype='low'):
    if 'JSBACH_S1_nbp' in instance_dict or 'JSBACH_S3_nbp' in instance_dict or 'S1' in instance_dict or 'S3' in instance_dict:
        vars = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']
    else:
        vars = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land',
                'Earth_Ocean', 'South_Ocean', 'North_Ocean', 'Tropical_Ocean']
    b_instance_dict = deepcopy(instance_dict)
    for model in instance_dict:
        b_instance_dict[model].data = xr.Dataset(
            {key: (('time'), instance_dict[model].bandpass(key, fc, fs=12, btype=btype)) for
            key in vars},
            coords={'time': (('time'), instance_dict[model].data.time.values)}
        )

    return b_instance_dict

""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"
TRENDY_MEAN_DIRECTORY = "./../../output/TRENDY/spatial/mean_all/"

year_invf = {}
month_invf = {}
for model in os.listdir(INV_DIRECTORY):
    model_dir = INV_DIRECTORY + model + '/'
    year_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
    month_invf[model] = invf.Analysis(xr.open_dataset(model_dir + 'month.nc'))

dmonth_invf = deseasonalise_instance(month_invf)

year_S1_trendy = {}
year_S3_trendy = {}
month_S1_trendy = {}
month_S3_trendy = {}
for model in os.listdir(TRENDY_DIRECTORY):
    model_dir = TRENDY_DIRECTORY + model + '/'

    if 'S1' in model:
        year_S1_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
        try:
            month_S1_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'month.nc'))
        except FileNotFoundError:
            pass
    elif 'S3' in model:
        year_S3_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'year.nc'))
        try:
            month_S3_trendy[model] = TRENDYf.Analysis(xr.open_dataset(model_dir + 'month.nc'))
        except FileNotFoundError:
            pass

year_mean = {
    "S1": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S1/year.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S3/year.nc'))
}
month_mean = {
    "S1": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S1/month.nc')),
    "S3": TRENDYf.Analysis(xr.open_dataset(TRENDY_MEAN_DIRECTORY + 'S3/month.nc'))
}

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]
