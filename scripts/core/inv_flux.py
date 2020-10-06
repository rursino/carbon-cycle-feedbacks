"""Perform outputs and analyses for raw inversion fluxes.
"""

""" IMPORTS"""
import numpy as np
import xarray as xr
import sys
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats, signal
from core import GCP_flux as GCPf
from itertools import *
from collections import namedtuple


""" INPUTS """
CURRENT_PATH = os.path.dirname(__file__)
MAIN_DIR = CURRENT_PATH + "./../../"


""" CLASSES """
class SpatialAgg:
    """This class takes an instance of the netCDF datasets from the
    data/inversions folder.
    It provides features to prepare the datasets for compatibility for the
    functions in the inv.Analysis class. This includes the integration of
    fluxes on spatial and temporal scales and conversion of cfdatetime time
    objects to datetime.

    Parameters
    ==========

    data: one of xarray.Dataset, xarray.DataArray and nc.file.

    """

    def __init__(self, data):
        """ Initialise an instance of an SpatialAgg. """
        if isinstance(data, xr.Dataset):
            _data = data

        elif type(data) == str and data.endswith('.pickle'):
            read_file = open(data, 'rb')
            _data = pickle.load(read_file)
            if not (isinstance(_data, xr.Dataset)):
                raise TypeError("Pickle object must be of\
                 type xr.Dataset or xr.DataArray.")

        else:
            _data = xr.open_dataset(data)

        self.data = _data
        self.earth_radius = 6.371e6 # Radius of Earth


    """The following three functions obtain the area of specific grid boxes of
    the Earth in different formats. It is used within the SpatialAgg class.
    """

    def scalar_earth_area(self, minlat, maxlat, minlon, maxlon):
        """Returns the area of earth in the defined grid box."""

        r_earth = self.earth_radius
        dtor = np.pi / 180. # conversion from degrees to radians.

        diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

        return 2. * np.pi * r_earth**2 * (np.sin(dtor * maxlat) -
        np.sin(dtor * minlat)) * (diff_lon[1] - diff_lon[0]) / 360.


    def earth_area(self, minlat, maxlat, minlon, maxlon):
        """Returns grid of areas with shape=(minlon, minlat) and earth area.
        """

        r_earth = self.earth_radius
        dtor = np.pi / 180. # conversion from degrees to radians.

        result = np.zeros((np.array(minlon).size, np.array(minlat).size))
        diffsinlat = 2. * np.pi * r_earth**2 *(np.sin(dtor*maxlat) -
        np.sin(dtor * minlat))
        diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

        for i in range(np.array(minlon).size):
            result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.

        return result

    def earth_area_grid(self, lats, lons):
        """Returns an array of the areas of each grid box within a defined set
        of lats and lons.
        """

        result = np.zeros((lats.size, lons.size))

        minlats = lats - 0.5*(lats[1] - lats[0])
        maxlats= lats + 0.5*(lats[1] - lats[0])
        minlons = lons - 0.5*(lons[1] - lons[0])
        maxlons= lons + 0.5*(lons[1] - lons[0])

        for i in range(lats.size):
            result[i,:] = self.scalar_earth_area(
            minlats[i],
            maxlats[i],
            minlons[0],
            maxlons[0]
            )

        return result

    def time_range(self, start_time, end_time, slice_obj=False):
        """ Returns a list or slice object of a range of time points, as is
        required when selecting time points for other functions.

        Parameters
        ==========

        start_time: string

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            If None is passed, then start_time refers to the first time point
            in the Dataset.

        end_time: string

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            If None is passed, then start_time refers to the last time point
            in the Dataset.

        slice_obj: bool, optional

            If True, then function returns a slice object instead of a list.
            Defaults to False.

        """

        df = self.data

        if start_time == None:
            start_time = df.time.values[0].strftime('%Y-%m')
        if end_time == None:
            end_time = df.time.values[-1]
            try:
                next_month = end_time.replace(month=end_time.month+1)
            except ValueError:
                next_month = end_time.replace(year=end_time.year+1, month=1)

            end_time = next_month.strftime('%Y-%m')

        arg_time_range = (
        pd
            .date_range(start=start_time, end=end_time, freq='M')
            .strftime('%Y-%m')
        )

        if slice_obj:
            return slice(arg_time_range[0], arg_time_range[-1])
        else:
            return arg_time_range

    def regional_cut(self, lats, lons, start_time=None, end_time=None):
        """ Cuts the dataset into selected latitude and longitude values.

        Parameters
        ==========

        lats: tuple-like

            tuple of start and end of latitudinal range to select in the cut.
            Must be of size two.

        lons: tuple-like

            tuple of start and end of latitudinal range to select in the cut.
            Must be of size two.

        start_time: string, optional

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            Default is None.

        end_time: string, optional

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            Default is None.

        """

        df = self.data

        arg_time_range = self.time_range(start_time, end_time)
        slice_time_range = self.time_range(start_time, end_time,
                                           slice_obj=True)

        df = df.sel(latitude=slice(*lats), longitude=slice(*lons),
                    time=slice_time_range)
        df = df * (self.earth_area_grid(df.latitude, df.longitude) * 1e-15)
        df = df.sum(axis=(1,2))
        df = df * (np.ones(len(df.time)) * 30/365)

        df['time'] = arg_time_range

        # Degree math symbol
        deg = u'\N{DEGREE SIGN}'
        df = df.assign_attrs(
                            {
                            "latitudes": "{}{} - {}{}".format(lats[0], deg,
                                                              lats[1], deg),
                            "longitudes": "{}{} - {}{}".format(lons[0], deg,
                                                               lons[1], deg)
                            }
                            )

        return df

    def latitudinal_splits(self, lat_split=30, start_time=None, end_time=None):
        """ Returns a xr.Dataset of the total global and regional carbon sink
        values at each time point within a range of time points.

        The regions are split into land and ocean and are latitudinally split
        according to passed argument for lat_split.
        This function uses the regional_cut function to cut the gridded carbon
        sinks to the latitudes required for the latitudinal splits.

        Globally integrated fluxes are also included for each of land and
        ocean.

        Parameters
        ==========

        lat_split: integer, optional

            Split the latitudes and output sums for each of those splits.
            If 23 is chosen, the latitudes are split by:
                90 degN to 23 degN, 23 degN to 23 degS and 23 degS to 90 degS.
            If 30 is chosen, the latitudes are split by:
                90 degN to 30 degN, 30 degN to 30 degS and 30 degS to 90 degS.
            And so on.

            Default is 30.

        start_time: string, optional

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            Default is None.

        end_time: string, optional

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            Default is None.

        """

        df = self.data

        vars = {
                "Earth": (-90, 90),
                "South": (-90, -lat_split),
                "Tropical": (-lat_split, lat_split),
                "North": (lat_split, 90)
                }

        values = {}
        for var in vars:
            region_df = self.regional_cut(vars[var], (-180,180),
                                    start_time=start_time, end_time=end_time)

            land_var = var + "_Land"
            land_region_vals = region_df["Terrestrial_flux"].values
            values[land_var] = land_region_vals

            ocean_var = var + "_Ocean"
            # Rayner has ocean variable as 'ocean' instead of 'Ocean_flux'.
            try:
                ocean_region_vals = region_df['Ocean_flux'].values
            except KeyError:
                ocean_region_vals = region_df['ocean'].values
            values[ocean_var] = ocean_region_vals

        slice_time_range = self.time_range(start_time, end_time, slice_obj=True)
        ds_time = df.sel(time=slice_time_range).time.values
        ds_time = [datetime.strptime(time.strftime('%Y-%m'), '%Y-%m')
                     for time in ds_time]

        ds = xr.Dataset(
            {key: (('time'), value) for (key, value) in values.items()},
            coords={'time': (('time'), ds_time)}
        )

        return ds


class Analysis:
    """ This class takes an instance of spatially and/or temporally integrated
    datasets and provides analysis and visualisations on the data as required,
    including plotting timeseries, linear regression over whole an decadal
    periods and frequency analyses.

    Parameters
    ==========

    data: an instance of an xarray.Dataset.

    """

    def __init__(self, data):
        """Take the xr.Dataset with cftime values and converts them into
        datetimes.
        """

        self.data = data
        _time = pd.to_datetime(data.time.values)

        # Extract time resolution
        timediff = _time[1] - _time[0]

        if timediff > pd.Timedelta('33 days'):
            self.time_resolution = "Y"
            self.tformat = "%Y"
        else:
            self.time_resolution = "M"
            self.tformat = "%Y-%m"

        CO2fname = MAIN_DIR + f"data/CO2/co2_year.csv"
        self.CO2 = pd.read_csv(CO2fname, index_col='Year')['CO2']

    def plot_timeseries(self, variable, time=None):
        """ Plot a variable against time.

        Parameters
        ==========

        variable:

            variable to plot against.

        time:

            select the time range for plot.
        """

        if time == None:
            time = slice(self.data.time.values[0], self.data.time.values[-1])

        df = self.data[variable].sel(time=time)

        plt.figure(figsize=(20,10))
        return plt.plot(df.time.values, df.values)

    def _time_to_CO2(self, time):
        """ Converts any time series array to corresponding atmospheric CO2
        values.
        This function should only be used within the 'cascading_window_trend'
        method.

        Parameters
        ==========

        time:

            time series to convert.

        """

        if self.time_resolution == "M":
            timeres = "month"
            index_col = ["Year", "Month"]
        else:
            timeres = "year"
            index_col = "Year"

        CO2 = self.CO2

        index = {
            "year": pd.to_datetime(time).year,
            "month": pd.to_datetime(time).month
        }

        if self.time_resolution == "M":
            index_to_pass = index["year"], index["month"]
        else:
            index_to_pass = index["year"]

        try:
            return CO2.loc[index_to_pass].values
        except AttributeError:
            return CO2.loc[index_to_pass]

    def cascading_window_trend(self, variable='Earth_Land', window_size=10, plot=False):
        """ Calculates the slope of the trend of an uptake variable for each
        time window and for a given window size. The function also plots the
        slopes as a timeseries and, if prompted, the r-value of each slope as
        a timeseries.

        Parameters
        ==========

        variable: string, optional

            carbon uptake variable to regress.
            Defaults to "Earth_Land".

        window_size: integer, optional

            size of time window of trends (in years).
            Defaults to 25.

        plot: bool, optional

            Option to show plots of the slopes.
            Defaults to False.
        """

        df = self.data

        index = pd.to_datetime(df.time.values).year

        cwt_df = pd.DataFrame(
            {
                'U': df[variable].values,
                'CO2': self.CO2.loc[index].values
            },
            index=index
        )

        cwt = []
        for i in range(len(cwt_df) - window_size):
            sub_df = cwt_df.iloc[i:i+window_size]
            slope = stats.linregress(sub_df.CO2.values, sub_df.U.values).slope
            cwt.append(slope)

        return pd.DataFrame({'CWT': cwt}, index=cwt_df.index[:-window_size]).CWT

    def psd(self, variable, fs, xlim=None, plot=False):
        """ Calculates the power spectral density (psd) of a timeseries of a
        variable using the Welch method. Also provides the timeseries plot and
        psd plot if passed. This function is designed for the monthly
        timeseries, but otherwise plotting the dataframe is possible.

        Parameters
        ==========

        variable: string

            carbon uptake variable to calculate psd.

        fs: integer

            sampling frequency.

        xlim: list-like

            apply limit to x-axis of the psd. Must be a list of two values.

        plot: bool, optional

            If assigned to True, shows two plots of timeseries and psd.
            Defaults to False.

        """

        df = self.data

        if self.time_resolution == "M":
            if fs == 12:
                period = " (years)"
                unit = "((GtC/yr)$^2$.yr)"
            elif fs == 1:
                period = " (months)"
                unit = "((GtC/yr)$^2$.month)"
            else:
                period = ""
                unit = ""
        else:
            if fs == 1:
                period = " (years)"
                unit = "((GtC/yr)$^2$.yr)"
            else:
                period = ""
                unit = ""

        freqs, spec = signal.welch(df[variable].values, fs=fs)

        if plot:
            plt.figure(figsize=(12,9))

            plt.subplot(211)
            plt.plot(df.time, df[variable].values)

            plt.subplot(212)
            plt.semilogy(1/freqs, spec)
            plt.gca().invert_xaxis()
            plt.xlim(xlim)

            plt.title(f"Power Spectrum of {variable}")
            plt.xlabel(f"Period{period}")
            plt.ylabel(f"Spectral Variance {unit}")

        return pd.DataFrame({f"Period{period}": 1/freqs,
        f"Spectral Variance {unit}": spec}, index=freqs)

    def deseasonalise(self, variable):
        """ Deseasonalise a timeseries of a variable by applying a low-pass
        filter with a 667-day cut-off period (as in Thoning (1989)).

        Parameters:
        ==========

        variable: string

            variable to apply from self object.

        """

        x = self.data[variable].values

        fs = 12
        fc = 365/667

        w = fc / (fs / 2) # Normalize the frequency.
        b, a = signal.butter(5, w, 'low')

        return signal.filtfilt(b, a, x)

    def bandpass(self, variable, fc, fs=1, order=5, btype="low"):
        """ Applies a bandpass filter to a dataset (either lowpass, highpass
        or bandpass) using the scipy.signal.butter function.

        Parameters:
        ===========

        variable: string

            variable to apply from self object.

        fc: integer

            cut-off frequency or frequencies.

        fs: integer

            sample frequency of x.

        order: integer, optional

            order of the filter.
            Defaults to 5.

        btype: string, optional

            options are low, high and band.
            Defaults to low.

        """

        x = self.data[variable].values

        if btype == "band":
            assert type(fc) == list, "fc must be a list of two values."
            fc = np.array(fc)

        w = fc / (fs / 2) # Normalize the frequency.
        b, a = signal.butter(order, w, btype)

        return signal.filtfilt(b, a, x)


class ModelEvaluation(Analysis):
    """ This class takes an instance of the model uptake datasets and provides
    evaluations against the Global Carbon Project (GCP) uptake timeseries.
    Must be annual resolution to match GCP.

    Parameters
    ==========

    data: xarray.Dataset

        dataset to use for model evaluation.

    """

    def __init__(self, data):
        """Initialise a ModelEvaluation instance with a xr.Dataset.
        """

        self.data = data

        _time = pd.to_datetime(data.time.values)

        start_year = _time.year[0]
        end_year = _time.year[-1]

        GCP = (pd
               .read_csv(MAIN_DIR + "data/GCP/budget.csv",
                         index_col=0,
                         usecols=[0,4,5,6]
                        )
               .loc[start_year:end_year]
              )

        GCP['CO2'] = pd.read_csv(MAIN_DIR + "data/CO2/co2_year.csv",
        index_col=0, header=0)[2:]
        GCP['land sink'] = GCP['land sink']
        GCP['ocean sink'] = GCP['ocean sink']
        GCP.rename(columns={"ocean sink": "ocean",
                            "land sink": "land"
                           },
                   inplace=True)

        GCP.index = _time
        self.GCP = GCP

        self.time_resolution = 'year'

    def regress_timeseries_to_GCP(self, sink, plot=None):
        """Calculates linear regression of model uptake to GCP uptake and
        shows a plot of the timeseries and scatter plot if requested.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        plot: bool, optional

            Plots timeseries and/or scatter plot of GCP and model uptake.
            None = No plots.
            timeseries = timeseries only.
            scatter = scatter plot only.
            both = both plots.
            Defaults to None.

        """

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        GCP = self.GCP

        linreg = stats.linregress(GCP[sink].values, df[model_sink].values)

        def timeseries_plot():
            plt.subplot(211).plot(GCP.index, GCP[sink])
            plt.subplot(211).plot(GCP.index, df[model_sink].values)
            plt.legend(["GCP", "Model"], fontsize=16)
            plt.xlabel("Year", fontsize=16)
            plt.ylabel("C flux to the atmosphere (GtC/yr)", fontsize=16)

        def scatter_plot():
            x = GCP[sink].values
            y = df[model_sink].values
            yy = linreg.slope * GCP[sink].values + linreg.intercept
            plt.subplot(212).scatter(x, y)
            plt.subplot(212).plot(x, yy, color='r')
            plt.xlabel("GCP (GtC/yr)", fontsize=16)
            plt.ylabel("Model (GtC/yr)", fontsize=16)

            xloc = 0.05 * (x.max() - x.min()) + x.min()
            yloc = 0.85 * (y.max() - y.min()) + y.min()
            text = (
                f'slope = {linreg.slope:.2f}\n'
                f'r = {linreg.rvalue:.2f}'
            )
            plt.text(xloc, yloc, text, fontsize=16)

        if plot:
            plt.figure(figsize=(20,12))

            if plot == "timeseries":
                timeseries_plot()
            elif plot == "scatter":
                scatter_plot()
            elif plot == "both":
                timeseries_plot()
                scatter_plot()

        return linreg

    def regress_cascading_window_trend_to_GCP(self, sink, window_size,
                                              indep="time", plot=None):
        """Calculates linear regression of model cascading window gradient to
        GCP cascading window gradient and shows a plot of the cascading
        gradients and scatter plot if requested.

        Parameters:
        ===========

        window_size: int

            size of window in years.

        sink: string

            either land or ocean.

        indep: string, optional

            Regress uptake variable over "time" or "CO2".
            Defaults to "time".

        plot: bool, optional

            Plots cascading window gradients and scatter plot of GCP.
            None = No plots.
            timeseries = timeseries only.
            scatter = scatter plot only.
            both = both plots.
            Defaults to None.

        """

        if "land" in sink:
            model_sink = "Earth_Land"
            GCP_sink = "land sink"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"
            GCP_sink = "ocean sink"

        df = self.data

        model_roll = (
        self
            .cascading_window_trend(indep=indep, variable = model_sink,
                                    window_size = window_size).values
            .squeeze()
        )

        GCP_roll_df = (
        GCPf
            .Analysis(GCP_sink)
            .cascading_window_trend(indep, window_size=window_size)
        )

        if indep == "CO2":
            index = self._time_to_CO2(self.GCP.index[:-window_size])
            GCP_roll = GCP_roll_df.loc[index].values.squeeze()
            xlabel = "CO2 (ppm)"
            cascading_yunit = "(GtC/ppm/yr)"
        elif indep == "time":
            index = self.GCP.index[:-window_size]
            GCP_roll = GCP_roll_df.loc[index.year].values.squeeze()
            xlabel = "Year"
            cascading_yunit = "(GtC/ppm$^2$)"

        linreg = stats.linregress(GCP_roll, model_roll)

        def timeseries_plot():
            plt.subplot(211).plot(index, GCP_roll)
            plt.subplot(211).plot(index, model_roll)
            plt.legend(["GCP", "model"], fontsize=16)
            plt.xlabel(xlabel, fontsize=16)
            plt.ylabel(f"Slope of C flux trend {cascading_yunit}", fontsize=16)

        def scatter_plot():
            x = GCP_roll
            y = model_roll
            yy = GCP_roll * linreg.slope + linreg.intercept
            plt.subplot(212).scatter(x, y)
            plt.subplot(212).plot(x, yy, color='r')
            plt.xlabel(f"GCP {cascading_yunit}", fontsize=16)
            plt.ylabel(f"Model {cascading_yunit}", fontsize=16)

            xloc = 0.05 * (x.max() - x.min()) + x.min()
            yloc = 0.85 * (y.max() - y.min()) + y.min()
            text = (
                f'slope = {linreg.slope:.2f}\n'
                f'r = {linreg.rvalue:.2f}'
            )
            plt.text(xloc, yloc, text, fontsize=16)

        if plot:
            plt.figure(figsize=(20,12))

            if plot == "timeseries":
                timeseries_plot()
            elif plot == "scatter":
                scatter_plot()
            elif plot == "both":
                timeseries_plot()
                scatter_plot()

        return linreg

    def compare_trend_to_GCP(self, sink, plot=False):
        """Calculates long-term trend of model uptake (over the whole time
        range) and GCP uptake. Also calculates the percentage difference of
        the trends.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        Returns:
        -----------

        Named tuple with following attributes:

            GCP_slope (MtC/yr)
            model_slope (MtC/yr)
            diff (%)

        """

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        GCP = self.GCP

        GCP_stats = stats.linregress(GCP.index.year, GCP[sink].values)
        model_stats = stats.linregress(GCP.index.year, df[model_sink].values)

        if plot:
            plt.bar(["GCP", "Model"], [GCP_stats[0], model_stats[0]])
            plt.ylabel("Trend (GtC/yr)", fontsize=14)

        trend_stats = namedtuple('Trend', ['GCP_slope', 'model_slope', 'diff'])

        return trend_stats(GCP_stats[0]*1e3, model_stats[0]*1e3,
                     GCP_stats[0]*100/model_stats[0]-100
                    )

    def autocorrelation_plot(self, variable):
        """Plots autocorrelation of model uptake timeseries using
        pandas.plotting.

        Parameters:
        ===========

        variable: string

            variable from self.data.

        """

        return pd.plotting.autocorrelation_plot(self.data[variable].values)
