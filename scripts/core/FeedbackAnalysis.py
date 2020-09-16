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
import os


""" INPUTS """
CURRENT_PATH = os.path.dirname(__file__)
MAIN_DIR = CURRENT_PATH + "./../../"


class INVF:
    def __init__(self, co2, temp, uptake, variable):
        """ Initialise an instance of the INVF FeedbackAnalysis class.

        Parameters:
        -----------

        co2: pd.Series

            co2 series. Must follow time resolution from 'uptake'.

        temp: xr.Dataset

            HadCRUT dataset. Must follow time resolution from 'uptake'.

        uptake: dict

            dictionary of inversion uptake datasets with different models.

        variable: str

            variable to regress in feedback analysis.

        """

        self.co2 = co2
        self.temp = temp
        self.uptake = uptake

        self.var = variable

        self.index = [(1980, 1989), (1992, 2001), (1990, 1999),
                      (2000, 2009), (2008, 2017)]

        self.time_periods = {
            "Rayner": ((1992, 2001), (2000, 2009)),
            "JENA_s76": ((1980, 1989), (1990, 1999), (2000, 2009), (2008, 2017)),
            "JENA_s85": ((1990, 1999), (2000, 2009), (2008, 2017)),
            "CTRACKER": ((2000, 2009), (2008, 2017)),
            "CAMS": ((1980, 1989), (1990, 1999), (2000, 2009), (2008, 2017)),
            "JAMSTEC": ((2000, 2009), (2008, 2017))
        }

        input_models = {}
        for model_name in self.uptake:
            model_time_periods = self.time_periods[model_name]
            start, end = model_time_periods[0][0], model_time_periods[-1][-1]
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

    def _feedback_model(self):
        """
        """

        input_models = self.input_models

        fb_models = {}
        for model_name in input_models:
            fb_models[model_name] = {}
            for start, end in self.time_periods[model_name]:
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

        params_dict = {'beta': {}, 'gamma': {}}
        for param in params_dict:
            if param == 'beta':
                p = 'C'
            elif param == 'gamma':
                p = 'T'
            for period in self.index:
                start, end = period
                if period not in params_dict[param].keys():
                    params_dict[param][start] = []
                for model_name in fb_models:
                    if period in fb_models[model_name]:
                        fb_model = fb_models[model_name][period]
                        params_dict[param][start].append(fb_model.params.loc[p])
                    else:
                        params_dict[param][start].append(np.nan)

        # Move the (1992, 2001) values to (1990, 1999)
        for param in params_dict:
            rayner_pos = np.where(~np.isnan(params_dict[param][1992]))[0].squeeze()
            rayner_val = params_dict[param][1992][rayner_pos]
            params_dict[param].pop(1992, None)
            params_dict[param][1990][5] = rayner_val


        params_df = {}
        for param in params_dict:
            df = pd.DataFrame(params_dict[param]).T
            df.columns = self.fb_models.keys()
            params_df[param] = df

        phi, rho = 0.015, 1.93
        params_df['beta'] /= 2.12
        params_df['u_gamma'] = params_df['gamma'] * phi / rho

        return params_df

    def regstats(self):
        """
        """

        fb_models = self.fb_models

        stats_dict = {}

        for model_name in fb_models:
            fb_model = fb_models[model_name]

            model_stats = {
                'Year': [],
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
                model_stats['Year'].append(time_period[0])
                model_stats['r_squared'].append(model.rsquared)
                model_stats['t_values_beta'].append(model.tvalues.loc['C'])
                model_stats['t_values_gamma'].append(model.tvalues.loc['T'])
                model_stats['p_values_beta'].append(model.pvalues.loc['C'])
                model_stats['p_values_gamma'].append(model.pvalues.loc['T'])
                model_stats['mse_total'].append(model.mse_total)
                model_stats['nobs'].append(model.nobs)

            df = pd.DataFrame(model_stats).set_index('Year')
            indices = [1980, 1990, 2000, 2008]
            for year in indices:
                if year not in df.index:
                    df.loc[year] = np.array([np.nan] * 7)
            if 1992 in df.index:
                df.loc[1990] = df.loc[1992]
                df.drop(1992, inplace=True)

            stats_dict[model_name] = df.sort_index()

        return stats_dict


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

        params_dict = {}

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

        params_dict['beta'] = pd.DataFrame(beta_dict).set_index('Year')
        params_dict['gamma'] = pd.DataFrame(gamma_dict).set_index('Year')

        phi, rho = 0.015, 1.93
        params_dict['beta'] /= 2.12
        params_dict['u_gamma'] = params_dict['gamma'] * phi / rho

        return params_dict

    def regstats(self):
        """
        """

        fb_models = self.fb_models

        stats_dict = {}

        for model_name in fb_models:
            fb_model = fb_models[model_name]

            model_stats = {
                'Year': [start for start, end in self.time_periods],
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

            stats_dict[model_name] = pd.DataFrame(model_stats).set_index('Year')

        return stats_dict
