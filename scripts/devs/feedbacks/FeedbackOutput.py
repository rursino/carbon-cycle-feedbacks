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
model_type = 'TRENDY_S1'
sink = 'Earth_Land'
timeres = 'month'

df = FeedbackAnalysis.FeedbackOutput(model_type, sink, timeres)

df.individual_plot('JSBACH_S1_nbp', 'beta')

df.merge_plot('const')
df.merge_plot('beta')
df.merge_plot('gamma')

df.merge_params('const')

df.stats_data
