""" Set of classes to establish results for the Airborne Fraction section.
"""

import numpy as np
import pandas as pd
import xarray as xr
import statsmodels.api as sm

import os
CURRENT_PATH = os.path.dirname(__file__)
MAIN_DIRECTORY = CURRENT_PATH + "./../../"

from core import FeedbackAnalysis


class GCP:
    def __init__(self, co2, temp):
        """ Initialise an instance of the AirborneFraction.GCP class.

        Parameters
        ----------

        co2: pd.Series

            co2 series. Must be the year time resolution.

        temp: xr.Dataset

            HadCRUT dataset. Must be the year time resolution.

        """

        self.co2 = co2[2:]
        self.temp = temp.sel(time=slice('1959', '2018')).Earth

        self.GCP = pd.read_csv(MAIN_DIRECTORY + 'data/GCP/budget.csv',
                               index_col='Year')

    def _feedback_parameters(self):
        """
        """

        models = {}
        for sink in ['land', 'ocean']:
            df = pd.DataFrame(data =
                {
                    "sink": self.GCP[sink + ' sink'],
                    "CO2": self.co2,
                    "temp": self.temp
                },
                index= self.GCP.index)

            X = sm.add_constant(df[['CO2', 'temp']])
            Y = df['sink']

            model = sm.OLS(Y, X).fit()

            models[sink] = model.params.loc[['CO2', 'temp']]

        return pd.DataFrame(models)

    def airborne_fraction(self, emission_rate=2):
        """
        """

        phi = 0.015 / 2.12
        rho = 1.93

        params = self._feedback_parameters().sum(axis=1)
        beta = params['CO2'] / 2.12
        u_gamma = params['temp'] * phi / rho

        b = 1 / np.log(1 + emission_rate / 100)
        u = 1 - b * (beta + u_gamma)

        return 1 / u


class INVF:
    def __init__(self, co2, temp, uptake):
        """ Initialise an instance of the AirborneFraction.INVF class.

        Parameters
        ----------

        co2: pd.Series

            co2 series. Must be in year time resolution.

        temp: xr.Dataset

            HadCRUT dataset. Must be in year time resolution.

        """

        self.co2 = co2
        self.temp = temp
        self.uptake = uptake

        self.phi = 0.015 / 2.12
        self.rho = 1.93

    def _feedback_parameters(self, variable):
        """
        """

        reg_models = {}

        model_names = self.uptake.keys()
        for model_name in model_names:
            U = self.uptake[model_name]

            start = int(U.time.values[0].astype('str')[:4])
            end = int(U.time.values[-1].astype('str')[:4])
            C = self.co2.loc[start:end]
            T = self.temp.sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T['Earth']
                                    }
                                   )

            X = sm.add_constant(df[["C", "T"]])
            Y = df["U"]

            reg_model = sm.OLS(Y, X).fit()

            reg_models[model_name] = reg_model.params[['C', 'T']].values

        reg_df = pd.DataFrame(reg_models, index=['beta', 'gamma'])
        reg_df.loc['u_gamma'] = reg_df.loc['gamma'] * self.phi / self.rho

        return reg_df

    def airborne_fraction(self, emission_rate=2):
        """
        """

        land = self._feedback_parameters('Earth_Land')
        ocean = self._feedback_parameters('Earth_Ocean')
        beta = (land + ocean).loc['beta'] / 2.12
        u_gamma = (land + ocean).loc['u_gamma']

        b = 1 / np.log(1 + emission_rate / 100)
        u = 1 - b * (beta + u_gamma)

        af = 1 / u
        return {'mean': af.mean(), 'std': af.std()}


class TRENDY:
    def __init__(self, co2, temp, uptake):
        """ Initialise an instance of the AirborneFraction.TRENDY class.
        Only S3 uptake should be used since there is no significance using S1
        in the analysis.

        Parameters
        ----------

        co2: pd.Series

            co2 series. Must be the year time resolution.

        temp: xr.Dataset

            HadCRUT dataset. Must be the year time resolution.

        """

        self.co2 = co2
        self.temp = temp
        self.uptake = uptake

        self.phi = 0.015 / 2.12
        self.rho = 1.93

    def _feedback_parameters(self, variable):
        """
        """

        start, end = 1960, 2017
        reg_models = {}

        model_names = self.uptake.keys()
        for model_name in model_names:
            U = self.uptake[model_name].sel(time=slice(str(start), str(end)))
            C = self.co2.loc[start:end]
            T = self.temp.sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T['Earth']
                                    }
                                   )

            X = sm.add_constant(df[["C", "T"]])
            Y = df["U"]

            reg_model = sm.OLS(Y, X).fit()

            reg_models[model_name] = reg_model.params[['C', 'T']].values

        reg_df = pd.DataFrame(reg_models, index=['beta', 'gamma'])
        reg_df.loc['beta'] /= 2.12
        reg_df.loc['u_gamma'] = reg_df.loc['gamma'] * self.phi / self.rho

        return reg_df

    def airborne_fraction(self, emission_rate=2):
        """
        """

        land = self._feedback_parameters('Earth_Land')

        ocean = GCP(self.co2, self.temp)._feedback_parameters()['ocean']
        ocean.index = ['beta', 'gamma']
        ocean['beta'] /= 2.12
        ocean['u_gamma'] = ocean['gamma'] * self.phi / self.rho

        params = land.add(ocean, axis=0)
        beta = params.loc['beta']
        u_gamma = params.loc['u_gamma']

        b = 1 / np.log(1 + emission_rate / 100)
        u = 1 - b * (beta + u_gamma)

        af = 1 / u
        return {'mean': af.mean(), 'std': af.std()}
