""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from statsmodels import api as sm

from copy import deepcopy

import os
from core import FeedbackAnalysis


""" INPUTS """
model_type = 'inversions'
sink = 'Earth_Land'
timeres = 'year'


df = FeedbackAnalysis.FeedbackOutput(model_type, sink, timeres)

inv_fb = df.params_data

inv_fb['CAMS'].loc[1979].values

merge_dataframe = {}
for model in inv_fb:
    model_df = inv_fb[model]
    for index in model_df.index:
        if index in merge_dataframe.keys():
            merge_dataframe[index].append(model_df.loc[index].values)
        else:
            merge_dataframe[index] = [model_df.loc[index].values]

np.array(merge_dataframe[2007]).mean(axis=0)

merge_mean = {}
for index in merge_dataframe:
    merge_mean[index] = np.array(merge_dataframe[index]).mean(axis=0)

inv_merge = pd.DataFrame(merge_mean).T.sort_index()

inv_merge

plt.bar(inv_merge.index, inv_merge[4])

df.individual_plot('CAMS', 'beta', True);
# df.merge_plot('S1', 'gamma', True);
# df.merge_params('S3', 'gamma')
# df.difference_simulations('gamma', False)

# phi, rho = 0.015, 1.93


""" FIGURES """
def fb_inv():
    merge_mean = dataframe.mean(axis=1)
    merge_std = dataframe.std(axis=1)

    if plot:
        plt.figure(figsize=(12,8))
        plt.bar(dataframe.index, merge_mean, yerr=merge_std)
        plt.xlabel("First year of multi-year window", fontsize = 16)
        plt.ylabel(f"{parameter.title()}", fontsize = 16)
        plt.title(f"{parameter.title()}: Multi-model mean", fontsize = 28)

    return {f'{parameter}': merge_mean, 'std': merge_std}


""" EXECUTION """
