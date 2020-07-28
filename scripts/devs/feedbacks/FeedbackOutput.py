""" Development for FeedbackOutput class functions.
"""

""" IMPORTS """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('./../../core')
import FeedbackAnalysis


""" INPUTS """
DIR = './../../../output/feedbacks/'

model = 'CLASS-CTEM_S1_nbp'
sink = 'Earth_Land'
temp = 'CRUTEM' if sink.split('_')[-1] == "Land" else "HadSST"
timeres = 'month'

params_fname = DIR + f'output_all/{model}/{temp}/{sink}/csv_output/params_{timeres}.csv'
stats_fname = DIR + f'output_all/{model}/{temp}/{sink}/csv_output/params_{timeres}.csv'

params = FeedbackAnalysis.FeedbackOutput(params_fname)
stats = FeedbackAnalysis.FeedbackOutput(stats_fname)

params.data
stats.data


def plot(param):
    lower = df.data[f'{param}_left'].values
    median = df.data[f'{param}_median'].values
    upper = df.data[f'{param}_right'].values

    std = np.sqrt(120) * (upper - lower) / 3.92

    plt.bar(df.data.index, df.data[f'{param}_median'], yerr=std)

plot('const')
plot('beta')
plot('gamma')
