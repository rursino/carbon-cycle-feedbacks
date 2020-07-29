""" Development for FeedbackOutput class functions.
"""

""" IMPORTS """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import sys
sys.path.append('./../../core')
import FeedbackAnalysis

from importlib import reload
reload(FeedbackAnalysis)


""" INPUTS """
DIR = './../../../output/feedbacks/'

model_type = 'TRENDY_S3'
sink = 'Earth_Land'
timeres = 'month'

df = FeedbackAnalysis.FeedbackOutput(model_type, sink, timeres)

# def individual_plot(model, parameter):
#     dataframe = df.params_data[model]
#
#     median = dataframe[f'{parameter}_median']
#     lower = dataframe[f'{parameter}_left']
#     upper = dataframe[f'{parameter}_right']
#
#     z = stats.norm.ppf(0.95)
#     std = (abs((upper - median) / z) + abs((lower - median) / z)) / 2
#
#     plt.bar(dataframe.index, dataframe[f'{parameter}_median'], yerr=std)

df.individual_plot('OCN_S3', 'beta')


# def merge_plot(parameter):
#
#     median = []
#     index = df.params_data[list(df.params_data)[0]].index
#
#     for model_name in df.params_data:
#         model = df.params_data[model_name]
#         median.append(model[f'{parameter}_median'].values)
#
#     dataframe = pd.DataFrame(median).T
#     dataframe.set_index(index, inplace=True)
#     dataframe.columns = df.models[df.models_type]
#
#     merge_mean = dataframe.mean(axis=1)
#     merge_std = dataframe.std(axis=1)
#     plt.bar(index, merge_mean, yerr=merge_std)

df.merge_plot('const')
df.merge_plot('beta')
df.merge_plot('gamma')
