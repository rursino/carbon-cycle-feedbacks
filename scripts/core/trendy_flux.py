"""Perform outputs and analyses for raw TRENDY terrestrial fluxes.
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
import GCP_flux as GCPf
from itertools import *


""" CLASSES """
class SpatialAgg:
    """This class takes an instance of the netCDF datasets from the
    data/TRENDY folder.
    It provides features to prepare the datasets for compatibility for the
    functions in the Analysis class. This includes the integration of
    fluxes on spatial and temporal scales and conversion of cfdatetime time
    objects to datetime.

    Parameters
    ==========

    data: xarray.Dataset or .nc file.

    """

    def __init__(self, data):
        """ Initialise an instance of an SpatialAgg. """
        if isinstance(data, xr.Dataset):
            _data = data

        elif type(data) == str and data.endswith('.pickle'):
            read_file = open(data, 'rb')
            _data = pickle.load(read_file)
            if not isinstance(_data, xr.Dataset):
                raise TypeError("Pickle object must be of type xr.Dataset.")

        else:
            _data = xr.open_dataset(data)

        self.var = 'nbp'
        self.data = _data[self.var]

        self.earth_radius = 6.371e6 # Radius of Earth

        # Metadata
        models_info_fname = os.path.join(os.path.dirname(__file__),
                            './../../data/TRENDY/models/models_info.txt')
        models_info = pd.read_csv(models_info_fname,
                                 delim_whitespace=True,
                                 index_col="File"
                                 )

        if isinstance(data, str):
            model_info = models_info.loc[data.split('/')[-1]]
            self.time_resolution = model_info['time_resolution']

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
        """Returns an array of the areas of each grid box within a defined set of
        lats and lons.
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

    def time_range(self, start_time=None, end_time=None, slice_obj=False):
        """ Returns a list or slice object of a range of time points, as is
        required when selecting time points for other functions.

        Parameters
        ==========

        start_time: string, optional

            The year of the time to start the integration in the format '%Y-%M'.
            If None is passed, then start_time refers to the first time point
            in the Dataset.
            Defaults to None.

        end_time: string, optional

            The year of the time to end the integration in the format '%Y-%M'.
            Note that the integration will stop at the year before argument.
            If None is passed, then end_time refers to the last time point
            in the Dataset.
            Defaults to None.

        slice_obj: bool, optional

            If True, then function returns a slice object instead of a list.
            Defaults to False.

        """

        df = self.data

        tformat = "%Y-%m" if self.time_resolution == "M" else "%Y"

        def format_time(time):

            return (pd
                        .to_datetime(datetime.strptime(
                        time.strftime(tformat), tformat)
                        )
                        .strftime(tformat)
                    )

        if start_time == None:
            try:
                start_time = format_time(df.time.values[0])
            except AttributeError:
                start_time = format_time(pd.to_datetime(df.time.values[0]))
        if end_time == None:
            try:
                end_time = format_time(df.time.values[-1])
            except AttributeError:
                end_time = format_time(pd.to_datetime(df.time.values[-1]))

        arg_time_range = (
        pd
            .date_range(start=start_time, end=end_time,
                        freq=self.time_resolution)
            .strftime(tformat)
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

            The year of the time to start the integration in the format '%Y-%M'.
            Default is None.

        end_time: string, optional

            The year of the time to end the integration in the format '%Y-%M'.
            Note that the integration will stop the month before argument.
            Default is None.

        """

        df = self.data

        arg_time_range = self.time_range(start_time, end_time)
        slice_time_range = self.time_range(start_time, end_time,
                                           slice_obj=True)

        df = df.sel(latitude=slice(*lats), longitude=slice(*lons),
                    time=slice_time_range)
        df = df * (self.earth_area_grid(df.latitude, df.longitude) * 1e-12)
        df = df.sum(axis=(1,2))
        df = df * np.ones(len(df.time)) * 30*24*3600

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
                "Earth_Land": (-90, 90),
                "South_Land": (-90, -lat_split),
                "Tropical_Land": (-lat_split, lat_split),
                "North_Land": (lat_split, 90)
                }

        values = {}
        for var in vars:
            region_df = self.regional_cut(vars[var], (-180,180),
                                    start_time=start_time, end_time=end_time)

            values[var] = region_df.values

        slice_time_range = self.time_range(start_time, end_time, slice_obj=True)
        ds_time = df.sel(time=slice_time_range).time.values

        ds = xr.Dataset(
            {key: (('time'), value) for (key, value) in values.items()},
            coords={'time': (('time'), ds_time)}
        )

        return ds

    def cftime_to_datetime(self, format='%Y-%m'):
        """Takes a xr.Dataset with cftime values and converts them into
        datetimes.

        Parameters
        ==========

        self: xr.Dataset.

        format: format of datetime.

        """

        time_list = []
        for time in self.data.time.values:
            time_value = datetime.strptime(time.strftime('%Y-%m'), '%Y-%m')
            time_list.append(time_value)

        return self.data.assign_coords(time=time_list)


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
            CO2_timeres = "year"
            CO2_index_col = "Year"

        else:
            self.time_resolution = "M"
            self.tformat = "%Y-%m"
            CO2_timeres = "month"
            CO2_index_col = ["Year", "Month"]

        CO2fname = f"./../../../data/CO2/co2_{CO2_timeres}.csv"
        self.CO2 = pd.read_csv(CO2fname, index_col=CO2_index_col)['CO2']

        if self.time_resolution == "M":
            slice_time = slice("{}-{}".format(*self.CO2.index[0]),
                               "{}-{}".format(*self.CO2.index[-1]))
        else:
            slice_time = slice("{}".format(self.CO2.index[0]),
                               "{}".format(self.CO2.index[-1]))
        self.CO2_time = self.data.time.sel(time=slice_time).values

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

        time: slice, optional

            time series to convert.

        """

        index = {
            "year": pd.to_datetime(time).year,
            "month": pd.to_datetime(time).month
        }

        if self.time_resolution == "M":
            index_to_pass = index["year"], index["month"]
        else:
            index_to_pass = index["year"]

        try:
            return self.CO2.loc[index_to_pass].values
        except AttributeError:
            return self.CO2.loc[index_to_pass]

    def cascading_window_trend(self, indep="time", variable="Earth_Land",
                            window_size=25, plot=False, include_pearson=False):
        """ Calculates the slope of the trend of an uptake variable for each
        time window and for a given window size. The function also plots the
        slopes as a timeseries and, if prompted, the r-value of each slope as
        a timeseries.

        Parameters
        ==========

        indep: string, optional

            Regress uptake variable over "time" or "CO2".
            Defaults to "time".

        variable: string, optional

            carbon uptake variable to regress.
            Defaults to "Earth_Land".

        window_size: integer, optional

            size of time window of trends (in years).
            Defaults to 25.

        plot: bool, optional

            Option to show plots of the slopes.
            Defaults to False.

        include_pearson: bool, optional

            Option to include dataframe and plot of
            r-values for each year.

            Defaults to False.

        """

        df = self.data

        def to_numerical(data):
            return (pd.to_numeric(data) /
                    (3600 * 24 * 365 *1e9) + 1969
                    )

        x_time = self.CO2_time
        if indep == "time":
            x_var = to_numerical(x_time)
            ts_xlabel = "Year"
            cascading_yunit = "(GtC/ppm/yr)"
            cascading_xlabel = f"First year of {window_size}-year window"
        else:
            x_var = self._time_to_CO2(x_time)
            ts_xlabel = "CO2 (ppm)"
            cascading_yunit = "(GtC/ppm$^2$)"
            cascading_xlabel = f"Start of {window_size}-year CO2 window (ppm)"

        if self.time_resolution == "M":
            window_size *= 12

        roll_vals, r_vals = [], []
        for i in range(0, len(x_time) - window_size):
            sub_x = x_var[i:i+window_size+1]
            sub_vals = df[variable].sel(time = slice(x_time[i],
                                               x_time[i+window_size])
                                       ).values

            linreg = stats.linregress(sub_x, sub_vals)

            roll_vals.append(linreg[0])
            r_vals.append(linreg[2])

        if self.time_resolution == "M":
            ws_plotlabel = f"{int(window_size / 12)}-year"
        else:
            ws_plotlabel = f"{window_size}-year"

        roll_df = pd.DataFrame({f"{ws_plotlabel} trend slope": roll_vals},
                               index=x_var[:-window_size]
                              )

        if plot:

            plt.figure(figsize=(22,16))

            plt.subplot(211)
            plt.plot(x_var, df.sel(time=self.CO2_time)[variable].values)
            plt.xlabel(ts_xlabel, fontsize=20)
            plt.ylabel("C flux to the atmosphere (GtC)", fontsize=20)

            plt.subplot(212)
            plt.plot(x_var[:-window_size], roll_df.values, color='g')
            plt.xlabel(cascading_xlabel, fontsize=20)
            plt.ylabel(f"Slope of C flux trend {cascading_yunit}", fontsize=20)

        if include_pearson:
            r_df = pd.DataFrame({"r-values of trends": r_vals},
                        index=x_var[:-window_size]
                        )
            return roll_df, r_df
        else:
            return roll_df

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
        """ Deseasonalise a timeseries of a variable by applying and using a
        seasonal index.

        Parameters:
        ==========

        variable: string

            variable to apply from self object.

        """

        x = self.data[variable].values

        mean_list = []
        for i in range(12):
            end_index = len(x)
            while True:
                indices = range(i, end_index + i, 12)
                try:
                    sub = x[indices]
                except IndexError:
                    end_index -= 12
                else:
                    break
            mean_list.append(np.mean(sub))

        i = 0
        s = []
        repeat = len(x) // len(mean_list)
        for i in range(repeat):
            for j in mean_list:
                s.append(j)
        last_indices = 0
        while len(s) < len(x):
            s.append(mean_list[last_indices])
            last_indices += 1

        return x - (s-np.mean(s))

    def bandpass(self, variable, fc, fs=1, order=5, btype="low",
    deseasonalise_first=False):
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

        deseasonalise_first: bool, optional

            Option to deseasonalise timeseries before applying bandpass filter.
            Defaults to False.

        """

        x = self.data[variable].values

        if deseasonalise_first:
            mean_list = []
            for i in range(12):
                end_index = len(x)
                while True:
                    indices = range(i, end_index + i, 12)
                    try:
                        sub = x[indices]
                    except IndexError:
                        end_index -= 12
                    else:
                        break
                mean_list.append(np.mean(sub))

            i = 0
            s = []
            repeat = len(x) // len(mean_list)
            for i in range(repeat):
                for j in mean_list:
                    s.append(j)
            last_indices = 0
            while len(s) < len(x):
                s.append(mean_list[last_indices])
                last_indices += 1

            x = x - (s-np.mean(s))

        if btype == "band":
            assert type(fc) == list, "fc must be a list of two values."
            fc = np.array(fc)

        w = fc / (fs / 2) # Normalize the frequency.
        b, a = signal.butter(order, w, btype)

        return signal.filtfilt(b, a, x)


class ModelEvaluation:
    """ This class takes an instance of the model uptake datasets and provides
    evaluations against the Global Carbon Project (GCP) uptake timeseries.
    Must be annual resolution to match GCP.

    Parameters
    ==========

    data: xarray.Dataset

        dataset to use for model evaluation.

    """

    def __init__(self, data):
        """Take the xr.Dataset with cftime values and converts them into
        datetimes.
        """

        self.data = data

        time_list = [datetime.strptime(time.strftime('%Y-%m'), '%Y-%m') for time in data.time.values]
        self.time = pd.to_datetime(time_list)


        start_year = self.time[0].year
        end_year = self.time[-1].year

        GCP = (pd
               .read_csv("./../../data/GCP/budget.csv",
                         index_col=0,
                         usecols=[0,4,5,6]
                        )
               .loc[start_year:end_year]
              )

        GCP['CO2'] = pd.read_csv("./../../data/CO2/co2_global.csv",
        index_col=0, header=0)[2:]
        GCP['land sink'] = GCP['land sink']
        GCP['ocean sink'] = GCP['ocean sink']
        GCP.rename(columns={"ocean sink": "ocean",
                            "land sink": "land"
                           },
                   inplace=True)

        self.GCP = GCP

    def plot_vs_GCP(self, sink, x="time"):
        """Plots variable chosen from model uptake timeseries and GCP uptake
        timeseries, either against time or CO2 concentration.

        Parameters:
        ===========
        sink: string

            either land or ocean.

        x: string, optional

            x axis; either time or CO2.
            Defaults to 'time'.

        """

        df = self.data
        GCP = self.GCP

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        plt.figure(figsize=(20,10))
        plt.ylabel("C flux to the atmosphere (GtC/yr)", fontsize=24)

        if x == "time":
            plt.plot(GCP.index, GCP[sink])
            plt.plot(GCP.index, df[model_sink].values) # FIX: Time needs to be integer on axes.
            plt.xlabel("Time", fontsize=22)

        elif x == "CO2":
            plt.plot(GCP.CO2.values, GCP[sink].values)
            plt.plot(GCP.CO2.values, df[model_sink].values)
            plt.xlabel("CO2 (ppm)", fontsize=22)

        else:
            raise ValueError("x must be 'time' or 'CO2'.")

        plt.legend(["GCP", "Model"], fontsize=20)

    def regress_timeseries_to_GCP(self, sink, plot=False):
        """Calculates linear regression of model uptake to GCP uptake and
        shows a plot of the timeseries and scatter plot if requested.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        plot: bool, optional

            Plots timeseries and scatter plot of GCP and model uptake if True.
            Defaults to False.

        """

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        GCP = self.GCP

        linreg = stats.linregress(GCP[sink].values, df[model_sink].values)

        if plot:
            plt.figure(figsize=(14,9))
            plt.subplot(211).plot(GCP.index, GCP[sink])
            plt.subplot(211).plot(GCP.index, df[model_sink].values)
            plt.legend(["GCP", "Model"], fontsize=16)
            plt.xlabel("Year", fontsize=16)
            plt.ylabel("C flux to the atmosphere (GtC/yr)", fontsize=16)

            plt.subplot(212).scatter(GCP[sink], df[model_sink].values)
            plt.xlabel("GCP (GtC/yr)", fontsize=16)
            plt.ylabel("Model (GtC/yr)", fontsize=16)

        return linreg

    def regress_cascading_window_trend_to_GCP(self, sink, window_size, plot=False):
        """Calculates linear regression of model cascading window gradient to
        GCP cascading window gradient and shows a plot of the cascading
        gradients and scatter plot if requested.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        plot: bool, optional

            Plots cascading window gradients and scatter plot of GCP and model
            uptake if True.
            Defaults to False.

        """


        def to_numeric(date):
            return date.year + (date.month-1 + date.day/31)/12


        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        time_range = self.GCP.index[:-window_size]

        model_roll = (
        Analysis
            .cascading_window_trend(self, model_sink, window_size).values
            .squeeze()
        )

        GCP_roll_df = (
        GCPf
            .Analysis("land sink")
            .cascading_window_trend(window_size=window_size)
        )
        GCP_roll = GCP_roll_df.loc[time_range].values.squeeze()


        # Plot
        if plot:
            plt.figure(figsize=(14,9))
            plt.subplot(211).plot(time_range, GCP_roll)
            plt.subplot(211).plot(time_range, model_roll)
            plt.legend(["GCP", "model"])
            plt.subplot(212).scatter(GCP_roll, model_roll)

        return stats.linregress(GCP_roll, model_roll)

    def compare_trend_to_GCP(self, sink, print_results=False):
        """Calculates long-term trend of model uptake (over the whole time
        range) and GCP uptake. Also calculates the percentage difference of
        the trends.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        """

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        GCP = self.GCP

        GCP_stats = stats.linregress(GCP.index, GCP[sink].values)
        model_stats = stats.linregress(GCP.index, df[model_sink].values)

        plt.bar(["GCP", "Model"], [GCP_stats[0], model_stats[0]])
        plt.ylabel("Trend (GtC/yr)", fontsize=14)

        if print_results:
            print(f"GCP slope: {GCP_stats[0]*1e3:.3f} MtC/yr",
                  f"Model slope: {model_stats[0]*1e3:.3f} MtC/yr",
                  f"Percentage difference: {((GCP_stats[0]*100/model_stats[0])-100):.2f}%", sep="\n")

        return {"GCP_slope (MtC/yr)": GCP_stats[0]*1e3,
                "Model_slope (MtC/yr)": model_stats[0]*1e3,
                "%_diff": (GCP_stats[0]*100/model_stats[0])-100}

    def autocorrelation_plot(self, variable):
        """Plots autocorrelation of model uptake timeseries using
        pandas.plotting.

        Parameters:
        ===========

        variable: string

            variable from self.data.

        """

        return pd.plotting.autocorrelation_plot(self.data[variable].values)
