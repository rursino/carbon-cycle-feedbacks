""" Developments to calculate model bias against GCP.

* Model evaluation of inversions with different time periods:
Complete (difference of two models) - (difference of GCP within the time
periods of two models)
e.g. [Model[JENA_s76] - Model[Rayner]] - [GCP[JENA_s76] - GCP[Rayner]]
"""

""" IMPORTS """
from core import inv_flux
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from itertools import *


""" INPUTS """
INPUT_DIRECTORY = './../../../output/inversions/spatial/output_all/'
MODELS = ['Rayner', 'CAMS', 'CTRACKER', 'JENA_s76', 'JENA_s85', 'JAMSTEC']

invf = {model:xr.open_dataset(INPUT_DIRECTORY + f'{model}/year.nc')[['Earth_Land', 'Earth_Ocean']]
for model in MODELS}

GCP = pd.read_csv('./../../../data/GCP/budget.csv', index_col='Year')[['ocean sink', 'land sink']]


def model_bias(models, sink):
    """ Calculate model bias based on formula above.

    Parameters:
    -----------

    models: list-like

        List of two models to use.

    sink: str

        Either land or ocean

    """

    models_df = {}

    for model in models:
        model_df = invf[model]
        start = pd.to_datetime(model_df.time.values[0]).year
        end = pd.to_datetime(model_df.time.values[-1]).year

        gcp_df = GCP.loc[start:end]

        models_df[model] = {'model': model_df[f'Earth_{sink.title()}'].mean(),
                            'GCP': gcp_df[f'{sink} sink'].mean()}

    return abs((models_df[models[1]]['model'] - models_df[models[0]]['model'])
                - (models_df[models[1]]['GCP'] - models_df[models[0]]['GCP'])).values


{models:model_bias(models, 'ocean') for models in combinations(MODELS, 2)}
