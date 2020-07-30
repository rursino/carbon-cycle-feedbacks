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
model_type = 'TRENDY_S3'
sink = 'Earth_Land'
timeres = 'month'

df = FeedbackAnalysis.FeedbackOutput(model_type, sink, timeres)

df.individual_plot('JSBACH_S3_nbp', 'beta', True)

df.merge_plot('const', False)
df.merge_plot('beta', True);
df.merge_plot('gamma', True);

df.merge_params('const')


# Pickle to P.Rayner for review.
import pickle
pickle.dump(df.merge_plot('const', False), open('merge_constant.pik', 'wb'))
pickle.dump(df.merge_plot('beta', False), open('merge_beta.pik', 'wb'))
pickle.dump(df.merge_plot('gamma', False), open('merge_gamma.pik', 'wb'))
