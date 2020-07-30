
""" This class facilitates the calculation of beta and gamma through a range of
spatial output datasets. It also includes further analysis and visualisations.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from datetime import datetime
import os


""" INPUTS """
CURRENT_PATH = os.path.dirname(__file__)


""" FUNCTIONS """
class FeedbackAnalysis:

    def __init__(self, uptake, temp, time='year', sink="Earth_Land",
                 time_range=None):
        """ Initialise a class for feedback analysis using three timeseries
        data: one each for uptake, CO2 and temperature.

        Parameters
        ==========

        uptake: list-like

            Information to pass a particular dataset.
            Should be passed as follows: (type, model)

        TEMP: str

            Name of the gridded temp. model to use.

        time: str, optional

            Time resolution to use. Pass either "year" or "month".
            Defaults to 'year'.

        sink: str, optional

            Latitudinal region to use over either land or ocean.
            Pass any of variables from the uptake and TEMP datasets.
            Defaults to 'Earth_Land'.

        time_range: slice, optional

            Selection of time range for feedabck analysis on the three datasets.
            Defaults to None, which passes the entire time range of the
            dataset.

        """

        DIR = os.path.join(CURRENT_PATH, './../../')
        OUTPUT_DIR = DIR + 'output/'
        self.OUTPUT_DIR = OUTPUT_DIR

        type, model = uptake
        Ufname = OUTPUT_DIR + f'{type}/spatial/output_all/{model}/{time}.nc'
        Cfname = DIR + f"data/CO2/co2_{time}.csv"
        Tfname = OUTPUT_DIR + f'TEMP/spatial/output_all/{temp}/{time}.nc'

        index_col = ["Year", "Month"] if time == "month" else ["Year"]
        C = pd.read_csv(Cfname, index_col=index_col)
        U = xr.open_dataset(Ufname)
        T = xr.open_dataset(Tfname)

        tformat = "%Y-%m" if time == "month" else "%Y"
        U_time = pd.to_datetime(U.time.values).strftime(tformat)
        T_time = pd.to_datetime(T.time.values).strftime(tformat)

        if time == "month":
            C_time = (pd
                        .to_datetime(["{}-{}".format(*time) for
                                                     time in C.index])
                        .strftime("%Y-%m")
                     )
            C.index = C_time

        else:
            C_time = (pd
                        .to_datetime([str(time) for time in C.index])
                        .strftime("%Y")
                     )

        intersection_index = (U_time
                                .intersection(C_time)
                                .intersection(T_time)
                             )

        if time_range == None:
            start = str(intersection_index[0])
            stop = str(intersection_index[-1])
            time_range = intersection_index
        else:
            start, stop = time_range.start, time_range.stop
            time_range = (pd
                    .date_range(
                        *(pd.to_datetime([start, stop])
                        ),
                        freq='Y')
                    .strftime("%Y")
                 )

        self.uptake = U.sel(time=slice(start, stop))[sink]
        self.temp = T.sel(time=slice(start, stop))[sink]
        self.CO2 = C.loc[start:stop].CO2

        df = pd.DataFrame(data = {
                                    "CO2": self.CO2.values,
                                    "Uptake": self.uptake.values,
                                    "Temp.": self.temp.values
                                 }
                         )

        self.data = df

        # Metadata
        self.uptake_type, self.uptake_model = type, model
        self.TEMP_model = temp
        self.time_res = time
        self.sink = sink
        self.start, self.stop = start, stop

        # Perform OLS regression on the three variables for analysis.
        X = self.data[["CO2", "Temp."]]
        Y = self.data["Uptake"]

        X = sm.add_constant(X)

        self.model = sm.OLS(Y, X).fit()
        self.model_summary = self.model.summary()

        # Attributes
        self.params = getattr(self.model, 'params')
        self.r_squared = getattr(self.model, 'rsquared')
        self.tvalues = getattr(self.model, 'tvalues')
        self.pvalues = getattr(self.model, 'pvalues')
        self.mse_total = getattr(self.model, 'mse_total')
        self.nobs = getattr(self.model, 'nobs')

    def plot(self):
        """ Plot all variables as a timeseries.
        """

        return plt.plot(self.data)

    def confidence_intervals(self, variable, alpha=0.05):

        return self.model.conf_int(alpha=alpha).loc[variable]

    def feedback_output(self, alpha=0.05):
        """ Extracts the relevant information from an OLS model (which should be
        created from the OLS_model method) and outputs it as a txt file in the
        output directory.

        """

        model = self.model

        confint_CO2 = self.confidence_intervals('CO2', alpha)
        confint_temp = self.confidence_intervals('Temp.', alpha)
        confint_const = self.confidence_intervals('const', alpha)

        line_break = '=' * 25 + '\n\n'
        writeLines = [
        "FEEDBACK ANALYSIS OUTPUT\n",
        line_break,
        f'Information\n' + '-' * 25 + '\n',
        f'Uptake Type: {self.uptake_type} \n',
        f'Uptake Model: {self.uptake_model} \n',
        f'Temperature Model: {self.TEMP_model} \n',
        f'Time Resolution: {self.time_res}\n',
        f"Time Range: {self.start} - {self.stop}\n",
        f'Sink: {self.sink} \n',
        line_break,
        'Results\n' + '-' * 25 + '\n\n',
        f"Parameter\t\t{alpha*100:.2f}%\t{(1-alpha)*100:.2f}%\n",
        '-' * 40 + '\n',
        f"Const    {self.params['const']:.3f}\t\t{confint_const[0]:.3f}\t{confint_const[1]:.3f}\n",
        f"BETA     {self.params['CO2']:.3f}\t\t{confint_CO2[0]:.3f}\t{confint_CO2[1]:.3f}\n",
        f"GAMMA    {self.params['Temp.']:.3f}\t\t{confint_temp[0]:.3f}\t{confint_temp[1]:.3f}",
        "\n\n",
        f'r: {np.sqrt(self.r_squared):.3f}  ',
        f'r^2: {self.r_squared:.3f}\n\n',
        f"\tConstant\tBETA\tGAMMA\n",
        '-' * 30 + '\n',
        f"t-value:\t{self.tvalues['const']:.3f}\t{self.tvalues['CO2']:.3f}\t{self.tvalues['Temp.']:.3f}\n",
        f"p-value:\t{self.pvalues['const']:.3f}\t{self.pvalues['CO2']:.3f}\t{self.pvalues['Temp.']:.3f}\n\n",
        f"MSE Total: {self.mse_total:.3f}\n\n",
        f"No. of observations: {self.nobs:.0f}\n",
        line_break
        ]

        OUTPUT_DIR = self.OUTPUT_DIR + 'feedbacks/'
        destination = f'{self.uptake_model}/{self.TEMP_model}/{self.sink}/'
        fname = (OUTPUT_DIR + destination +
                 f'{self.start}_{self.stop}.txt')

        try:
            with open(fname, 'w') as f:
                f.writelines(writeLines)
        except FileNotFoundError:
            os.makedirs(OUTPUT_DIR + destination)
            with open(fname, 'w') as f:
                f.writelines(writeLines)


class FeedbackOutput:

    def __init__(self, model_type, sink, timeres):
        """ Initialise a FeedbackOutput instance by reading a csv file from
        csv feedback outputs.
        """

        models = {
            'inversions': ['CAMS', 'JENA_s76', 'JENA_s85',
                           'JAMSTEC', 'Rayner', 'CTRACKER'],
            'TRENDY_S1': ['CABLE-POP_S1_nbp', 'CLASS-CTEM_S1_nbp',
                          'JSBACH_S1_nbp', 'LPJ-GUESS_S1_nbp', 'OCN_S1_nbp'],
            'TRENDY_S3': ['CABLE-POP_S3_nbp', 'CLASS-CTEM_S3_nbp',
                          'JSBACH_S3_nbp', 'LPJ-GUESS_S3_nbp', 'OCN_S3_nbp']
            }

        temp = 'CRUTEM' if sink.split('_')[-1] == "Land" else "HadSST"

        DIR = CURRENT_PATH + './../../output/feedbacks/'

        self.params_data = {}
        self.stats_data = {}
        for model in models[model_type]:
            model_name = model if model_type == 'inversions' else model + 'nbp'
            FNAME_DIR = DIR + f'output_all/{model}/{temp}/{sink}/csv_output/'

            params_fname = FNAME_DIR + f'params_{timeres}.csv'
            stats_fname = FNAME_DIR + f'stats_{timeres}.csv'

            self.params_data[model] = pd.read_csv(params_fname, index_col = 0)
            self.stats_data[model] = pd.read_csv(stats_fname, index_col = 0)

        self.models = models
        self.models_type = model_type
        self.sink = sink
        self.timeres = timeres


    def merge_params(self, parameter):
        """ Merge all parameter median values all models in the specified model
        type.
        NOTE: Only works for TRENDY datasets since time range is consistent.
        """

        median = [
                self.params_data[model_name][f'{parameter}_median'].values for
                model_name in self.params_data
            ]

        dataframe = pd.DataFrame(median).T
        dataframe.set_index(self.params_data[list(self.params_data)[0]].index,
                            inplace=True)
        dataframe.columns = self.models[self.models_type]

        return dataframe

    def individual_plot(self, model, parameter, plot=False):
        """ Create a bar plot of the chosen model and parameter. The standard
        deviation is calculated using the upper and lower limits of the
        confidence intervals of the parameter.

        Parameters
        ==========

        model: string

            chosen model from the self.model_type.

        parameter: string

            one of "const", "beta", "gamma"

        plot: bool, optional

            Show bar plot of parameter values if True.
            Defaults to False.
        """

        dataframe = self.params_data[model]

        median = dataframe[f'{parameter}_median']
        lower = dataframe[f'{parameter}_left']
        upper = dataframe[f'{parameter}_right']

        z = stats.norm.ppf(0.95)
        std = (abs((upper - median) / z) + abs((lower - median) / z)) / 2

        if plot:
            plt.figure(figsize=(12,8))
            plt.bar(dataframe.index, median, yerr=std)
            plt.xlabel("First year of multi-year window", fontsize = 16)
            plt.ylabel(f"{parameter.title()} (90% CI)", fontsize = 16)
            plt.title(f"{parameter.title()}: {model}", fontsize = 28)


        return {f'{parameter}': median, 'std': std}

    def merge_plot(self, parameter, plot=False):
        """ Create a bar plot of the chosen parameter. The values are
        calculated by taking the average of all models. The standard deviation
        is also calculated by taking the standard deviation of the set of values
        from all models.
        NOTE: Only works for TRENDY datasets since time range is consistent.

        Parameters
        ==========

        parameter: string

            one of "const", "beta", "gamma"

        plot: bool, optional

            Show bar plot of parameter values if True.
            Defaults to False.
        """

        dataframe = self.merge_params(parameter)

        merge_mean = dataframe.mean(axis=1)
        merge_std = dataframe.std(axis=1)

        if plot:
            plt.figure(figsize=(12,8))
            plt.bar(dataframe.index, merge_mean, yerr=merge_std)
            plt.xlabel("First year of multi-year window", fontsize = 16)
            plt.ylabel(f"{parameter.title()} (90% CI)", fontsize = 16)
            plt.title(f"{parameter.title()}: Multi-model mean", fontsize = 28)
        
        return {f'{parameter}': merge_mean, 'std': merge_std}
