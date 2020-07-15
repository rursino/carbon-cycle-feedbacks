""" This class facilitates the calculation of beta and gamma through a range of
spatial output datasets. It also includes further analysis and visualisations.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import os


""" FUNCTIONS """
class FeedbackAnalysis:

    def __init__(self, uptake, temp, CO2="weighted", time='year',
    sink="Earth_Land", time_range=None):
        """ Initialise a class for feedback analysis using three timeseries
        data: one each for uptake, CO2 and temperature.

        Parameters
        ==========

        uptake: list-like

            Information to pass a particular dataset.
            Should be passed as follows: (type, model)

        CO2: str

            Information to pass a particular csv file.

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

        DIR = './../../'
        OUTPUT_DIR = DIR + 'output/'
        self.OUTPUT_DIR = OUTPUT_DIR

        type, model = uptake
        Ufname = OUTPUT_DIR + f'{type}/spatial/output_all/{model}/{time}.nc'
        Cfname = DIR + f"data/CO2/co2_{time}_{CO2}.csv"
        Tfname = OUTPUT_DIR + f'TEMP/spatial/output_all/{temp}/{time}.nc'

        U = xr.open_dataset(Ufname)
        C = pd.read_csv(Cfname)
        T = xr.open_dataset(Tfname)

        tformat = "%Y-%m" if time == "month" else "%Y"
        U_time = (pd
                    .to_datetime([datetime.strptime(
                    time.strftime(tformat), tformat) for time in U.time.values]
                    )
                    .strftime(tformat)
                 )
        T_time = pd.to_datetime(T.time.values).strftime(tformat)

        if time == "month":
            CO2timezip = zip(C["Year"].values, C["Month"].values)

            CO2time = []
            for time in CO2timezip:
                year = str(time[0])
                month = f"0{time[1]}" if time[1] < 10 else str(time[1])
                CO2time.append(year+'-'+month)

            C_time = pd.to_datetime(CO2time).strftime("%Y-%m")
            C.index = C_time

        else:
            C_time = (pd
                        .to_datetime([str(year) for year in C["Year"].values])
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
                        *(pd.to_datetime([start, stop]) + pd.offsets.YearEnd()
                        ),
                        freq='Y')
                    .strftime("%Y")
                 )

        self.uptake = U.sel(time=slice(start, stop))[sink]
        self.temp = T.sel(time=slice(start, stop))["Earth"] #sink
        self.CO2 = C.loc[start:stop].CO2

        df = pd.DataFrame(data = {
                                    "CO2": self.CO2.values,
                                    "Uptake": self.uptake.values,
                                    "Temp.": self.temp.values
                                 },
                          index = time_range
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
        f"Parameter\t\t\t{alpha*100:.2f}%\t{(1-alpha)*100:.2f}%\n",
        '-' * 40 + '\n',
        f"BETA\t{self.params['CO2']:.3f}\t\t{confint_CO2[0]:.3f}\t{confint_CO2[1]:.3f}\n",
        f"GAMMA\t{self.params['Temp.']:.3f}\t\t{confint_temp[0]:.3f}\t{confint_temp[1]:.3f}",
        "\n\n",
        f'r: {np.sqrt(self.r_squared):.3f}, ',
        f'r^2: {self.r_squared:.3f}\n\n',
        f"\t\t\tBETA\tGAMMA\n",
        '-' * 30 + '\n',
        f"t-value:\t{self.tvalues['CO2']:.3f}\t{self.tvalues['Temp.']:.3f}\n",
        f"p-value:\t{self.pvalues['CO2']:.3f}\t{self.pvalues['Temp.']:.3f}\n\n",
        f"MSE Total: {self.mse_total:.3f}\n\n",
        f"No. of observations: {self.nobs:.0f}\n",
        line_break
        ]

        OUTPUT_DIR = self.OUTPUT_DIR + 'feedbacks/'
        destination = f'{self.uptake_model}/{self.TEMP_model}/{self.start}-{self.stop}.txt'
        fname = OUTPUT_DIR + destination

        try:
            with open(fname, 'w') as f:
                f.writelines(writeLines)
        except FileNotFoundError:
            os.makedirs(OUTPUT_DIR + f'{self.uptake_model}/{self.TEMP_model}/')
            with open(fname, 'w') as f:
                f.writelines(writeLines)


class FeedbackOutput:
    def __init__():
        pass
