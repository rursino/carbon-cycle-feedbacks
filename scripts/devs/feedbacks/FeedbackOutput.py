""" Development for FeedbackOutput class functions.
"""

""" IMPORTS """
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.dates import date2num

import sys
from core import FeedbackAnalysis

from importlib import reload
reload(FeedbackAnalysis)


""" INPUTS """
model_type = 'TRENDY'
sink = 'South_Land'
timeres = 'year'


""" EXECUTION """
df = FeedbackAnalysis.FeedbackOutput(model_type, sink, timeres)

df.individual_plot('JSBACH_S1_nbp', 'beta', True);
df.merge_plot('S1', 'beta', True);
df.merge_params('S3', 'gamma')
df.difference_simulations('const', False)
