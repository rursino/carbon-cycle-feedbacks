""" Calculation of beta and gamma with methodology via methods.txt.
"""

""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime

import sys
sys.path.append("./../../core")
import FeedbackAnalysis as FA

from importlib import reload
reload(FA)

df = FA.FeedbackAnalysis(uptake=('inversions', 'CAMS'), temp='CRUTEM',
                        time='year', CO2='raw', sink="Earth_Land",
                        time_range = None)

df.model.summary()
df.confidence_intervals('CO2', 0.05)
df.feedback_output(OUTPUT_DIR + 'feedbacks/test.txt')


# FOR OUTPUT: MORE INFO

# METHODS
#'predict'
#'save'
#'summary'
#'wald_test'
#'wald_test_terms'

# UNSUITABLE ATTRS
#'resid'
#'wresid'
