""" Development for FeedbackOutput class functions.
"""

""" IMPORTS """
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.append('./../../core')
import FeedbackAnalysis


""" INPUTS """
DIR = './../../../output/feedbacks/'

model = 'CAMS' #'CABLE-POP_S1_nbp'
sink = 'South_Ocean'
temp = 'CRUTEM' if sink.split('_')[-1] == "Land" else "HadSST"
df_fname = 'params_month'

fname = DIR + f'output_all/{model}/{temp}/{sink}/csv_output/{df_fname}.csv'

df = FeedbackAnalysis.FeedbackOutput(fname)

df.data
