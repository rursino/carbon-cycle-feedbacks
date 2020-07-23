""" Calculation of beta and gamma with methodology via methods.txt.
"""

""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import pandas as pd

import sys
sys.path.append("./../../core")
import FeedbackAnalysis as FA

from importlib import reload
reload(FA);


df = FA.FeedbackAnalysis(uptake=('TRENDY', 'LPJ-GUESS_S1_nbp'), temp='CRUTEM',
                        time='year', sink="Earth_Land",
                        time_range = slice("2008", "2017"))
df.data

df.U
len(df.uptake), len(df.temp), len(df.CO2)




df.data
df.model.summary()

df.data

df.model.params
df.confidence_intervals('const', alpha)
df.model.summary()
df.confidence_intervals('CO2', 0.05)

df.feedback_output()


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
