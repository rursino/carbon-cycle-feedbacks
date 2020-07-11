""" Calculation of beta and gamma with methodology via methods.txt.
"""

""" IMPORTS """

import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


""" INPUTS """
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'

# Uptake - inversions
U = xr.open_dataset(OUTPUT_DIR + 'inversions/spatial/output_all/CAMS/year.nc')
C = pd.read_csv(DIR + 'data/CO2/co2_global.csv', index_col='Year')['CO2']
T = xr.open_dataset(OUTPUT_DIR + 'TEMP/spatial/output_all/CRUTEM/year.nc')

class FeedbackAnalysis:

    def __init__(self, uptake, CO2, temp):

        U_time = (pd
                    .to_datetime([time.year for time in uptake.time.values],
                                format='%Y'
                                )
                    .year
                 )
        C_time = CO2.index
        T_time = pd.to_datetime(temp.time.values).year

        intersection_index = (U_time
                                .intersection(C_time)
                                .intersection(T_time)
                             )
        time_range = slice(str(intersection_index[0]),
                           str(intersection_index[-1])
                          )

        self.uptake = uptake.sel(time=time_range)
        self.temp = temp.sel(time=time_range)
        self.CO2 = CO2.loc[time_range]

        df = pd.DataFrame(data = {
                                    "CO2": self.CO2.values,
                                    "Uptake": self.uptake.Earth_Land.values,
                                    "Temp.": self.temp.Earth.values
                                 },
                          index = intersection_index
                         )

        self.data = df

    def plot(self):
        """ Plot all variables as a timeseries.
        """

        return plt.plot(self.data)

    def OLS_model(self):
        """ Converts uptake, temperature and CO2 data into a single pandas
        dataframe and performs an OLS regression on the three variables.

        """

        df = self.data

        X = df[["CO2", "Temp."]]
        Y = df["Uptake"]

        return sm.OLS(Y, X).fit()

    def feedback_output(self, model, fname):
        """ Extracts the relevant information from an OLS model (which should be
        created from the OLS_model method) and outputs it as a txt file in the
        output directory.

        """

        with open(fname, 'w') as f:
            line_break = '=' * 25 + '\n\n'

            f.write('TEST OUTPUT\n' + line_break)
            f.write(f'Information\n' + '-' * 25 + '\n')
            f.write(f'Model: _ \n')
            # f.write()


df = FeedbackAnalysis(uptake=U, CO2=C, temp=T)

df.feedback_output(df.OLS_model(), OUTPUT_DIR + 'feedbacks/test.txt')

'conf_int, alpha: 0.05'
'mse_total'
'nobs'
'params'
'predict'
'pvalues'
'resid'
'rsquared'
'save'
'summary'
'tvalues'
'wald_test'
'wald_test_terms'
'wresid'













def time_intersection(*args):
    """ Grabs a set of pd.DatetimeIndex objects and returns the intersection of
    time points as a slice object. This makes indexing dataframes and datasets
    to match time points easily accessible.
    Note that slice obj represents each date as strings.

    Parameters
    ==========

    *args: pd.DateTimeIndex

        A set of arguments to pass pd.DateTimeIndex objects for matching to
        other objects.

    """

    intersection_index = args[0]
    for index in args[1:]:
         intersection_index = intersection_index.intersection(index)

    return slice(str(intersection_index[0]), str(intersection_index[-1]))

intersection_index = time_intersection(C_time, U_time, U_time)
U = U.sel(time=intersection_index)
T = T.sel(time=intersection_index)
C = C.loc[intersection_index]

C.values

def OLS_model(U, C, T):
    """ Converts uptake, temperature and CO2 data into a single pandas dataframe
    and performs an OLS regression on the three variables.

    U, C and T must have the same length and contain indentical time ranges,
    which can be acheived by calling the time_intersection method.

    Parameters
    ==========

    U: xr.Dataset

        uptake data.

    C: pd.Series

        CO2 concentrations data.

    T: xr.Dataset

        temperature data.

    """

    X = df[["CO2", "Temp."]]
    Y = df["Uptake"]

    return sm.OLS(Y, X).fit()

OLS_model(U, C, T).summary()

def feedback_output(model):
    """ Extracts the relevant information from an OLS model (which should be
    created from the OLS_model method) and outputs it as a txt file in the
    output directory.

    """

    with open(OUTPUT_DIR + 'feedbacks/test.txt', 'w') as f:
        line_break = '=' * 25 + '\n'
        f.write('TEST OUTPUT\n' + line_break)
        f.write(f'Model: _ \n' + line_break)
        f.write(f'Information')
