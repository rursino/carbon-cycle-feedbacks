
""" This class facilitates the calculation of beta and gamma through a range of
spatial output datasets. It also includes further analysis and visualisations.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels import api as sm
from scipy import stats
from scipy import signal
from datetime import datetime
import os
# from copy import deepcopy


""" INPUTS """
CURRENT_PATH = os.path.dirname(__file__)
MAIN_DIR = CURRENT_PATH + "./../../"


""" FUNCTIONS """
class TRENDY:
    def __init__(self, co2, temp, uptake, variable):
        """ Initialise an instance of the TRENDY FeedbackAnalysis class.

        Parameters:
        -----------

        co2: pd.Series

            co2 series. Must follow time resolution from 'uptake'.

        temp: xr.Dataset

            HadCRUT dataset. Must follow time resolution from 'uptake'.

        uptake: dict

            dictionary of TRENDY uptake datasets with different models.

        variable: str

            variable to regress in feedback analysis.

        """

        self.co2 = co2
        self.temp = temp
        self.uptake = uptake

        self.var = variable

        self.time_periods = (
            (1960, 1969),
            (1970, 1979),
            (1980, 1989),
            (1990, 1999),
            (2000, 2009),
            (2008, 2017),
        )

        # Pre-processing of data
        self.timedelta = (
                        (self.uptake["OCN"].time[1] - self.uptake["OCN"].time[0])
                            .values
                            .astype('timedelta64[M]')
                    )
        if self.timedelta == np.timedelta64('1', 'M'):
            # Deseasonalise monthly data
            for model in uptake:
                self.uptake[model] = xr.Dataset(
                    {key: (('time'), self._deseasonalise(model, key)) for
                    key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']},
                    coords={'time': (('time'), self.uptake[model].time.values)}
                )

        start, end = 1960, 2017
        input_models = {}
        for model_name in self.uptake:
            C = self.co2.loc[start:end]
            T = self.temp.sel(time=slice(str(start), str(end)))
            U = self.uptake[model_name].sel(time=slice(str(start), str(end)))

            input_models[model_name] = pd.DataFrame(data = {
                                                "C": C,
                                                "U": U[self.var],
                                                "T": T[self.var.split('_')[0]]
                                                }
                                               )

        self.input_models = input_models
        self.fb_models = self._feedback_model()


    def _deseasonalise(self, model, variable):
        """
        """

        x = self.uptake[model][variable].values

        fs = 12
        fc = 365/667

        w = fc / (fs / 2) # Normalize the frequency.
        b, a = signal.butter(5, w, 'low')

        return signal.filtfilt(b, a, x)

    def _feedback_model(self):
        """
        """

        input_models = self.input_models

        fb_models = {}
        for model_name in input_models:
            fb_models[model_name] = {}

            for start, end in self.time_periods:
                # Perform OLS regression on the three variables for analysis.
                X = input_models[model_name][["C", "T"]].loc[start:end]
                Y = input_models[model_name]["U"].loc[start:end]
                X = sm.add_constant(X)

                fb_models[model_name][(start, end)] = sm.OLS(Y, X).fit()

        return fb_models

    def params(self):
        """
        """

        fb_models = self.fb_models

        beta_dict = {'Year': [start for start, end in self.time_periods]}
        gamma_dict = {'Year': [start for start, end in self.time_periods]}
        for model_name in fb_models:
            fb_model = fb_models[model_name]

            params = {'beta': [], 'gamma': []}

            for time_period in fb_model:
                model = fb_model[time_period]

                # Attributes
                params['beta'].append(model.params.loc['C'])
                params['gamma'].append(model.params.loc['T'])

            beta_dict[model_name] = params['beta']
            gamma_dict[model_name] = params['gamma']

        betaDF = pd.DataFrame(beta_dict).set_index('Year')
        gammaDF = pd.DataFrame(gamma_dict).set_index('Year')

        phi, rho = 0.015, 1.93
        betaDF /= 2.12
        gammaDF *= phi / rho

        return betaDF, gammaDF

    def regstats(self):
        """
        """

        fb_models = self.fb_models

        stats_dict = {
            'Year': [start for start, end in self.time_periods]
        }

        for model_name in fb_models:
            fb_model = fb_models[model_name]

            model_stats = {
                'r_squared': [],
                't_values_beta': [],
                't_values_gamma': [],
                'p_values_beta': [],
                'p_values_gamma': [],
                'mse_total': [],
                'nobs': []
            }

            for time_period in fb_model:
                model = fb_model[time_period]

                # Attributes
                model_stats['r_squared'].append(model.rsquared)
                model_stats['t_values_beta'].append(model.tvalues.loc['C'])
                model_stats['t_values_gamma'].append(model.tvalues.loc['T'])
                model_stats['p_values_beta'].append(model.pvalues.loc['C'])
                model_stats['p_values_gamma'].append(model.pvalues.loc['T'])
                model_stats['mse_total'].append(model.mse_total)
                model_stats['nobs'].append(model.nobs)

            stats_dict[model_name] = model_stats

        return stats_dict
