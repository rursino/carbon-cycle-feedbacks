"""
"""

""" IMPORTS"""
import numpy as np
import xarray as xr
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats, signal
import GCP_flux as GCPf
from itertools import *


"""The following three functions obtain the area of specific grid boxes of the
Earth in different formats. It is used in the spatial_integration function
within the SpatialAgg class.
"""

from numpy import pi
r_earth = 6.371e6 # Radius of Earth
dtor = pi/180. # conversion from degrees to radians.

def scalar_earth_area(minlat, maxlat, minlon, maxlon):
    """Returns the area of earth in the defined grid box."""

    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)

    return 2.*pi*r_earth**2 *(np.sin(dtor*maxlat) -
    np.sin(dtor*minlat))*(diff_lon[1]-diff_lon[0])/360.


def earth_area(minlat, maxlat, minlon, maxlon):
    """Returns grid of areas with shape=(minlon, minlat) and earth area.
    """

    result = np.zeros((np.array(minlon).size, np.array(minlat).size))
    diffsinlat = 2.*pi*r_earth**2 *(np.sin(dtor*maxlat) - np.sin(dtor*minlat))
    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)

    for i in range(np.array(minlon).size):
        result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.

    return result

def earth_area_grid(lats,lons):
    """Returns an array of the areas of each grid box within a defined set of
    lats and lons.
    """

    result=np.zeros((lats.size, lons.size))

    minlats = lats - 0.5*(lats[1]-lats[0])
    maxlats= lats + 0.5*(lats[1]-lats[0])
    minlons = lons - 0.5*(lons[1]-lons[0])
    maxlons=lons + 0.5*(lons[1]-lons[0])

    for i in range(lats.size):
        result[i,:] = scalar_earth_area(
        minlats[i],
        maxlats[i],
        minlons[0],
        maxlons[0]
        )

    return result




class SpatialAgg:
    """This class takes an instance of the netCDF datasets from the
    Peylin_flux/data folder.
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
        if isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
            _data = data

        elif type(data) == str and data.endswith('.pickle'):
            read_file = open(data, 'rb')
            _data = pickle.load(read_file)
            if not (isinstance(_data, xr.Dataset) or isinstance(_data, xr.DataArray)):
                raise TypeError("Pickle object must be of type xr.Dataset or xr.DataArray.")

        else:
            _data = xr.open_dataset(data)

        self.data = _data

    def spatial_integration(self, start_time=None, end_time=None,
    lat_split=30, lon_range=None):
        """Returns a xr.Dataset of total global and regional sinks at a
        specific time point or a range of time points.
        The regions are split into land and ocean and are latitudinally split
        according to passed argument for lat_split.
        A longitudinal range can also be chosen.
        Globally integrated fluxes are also included for each of land and ocean.

        Parameters
        ==========

        start_time: string, optional

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            Default is None.

        end_time: string, optional

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            Default is None.

        lat_split: integer, optional

            Split the latitudes and output sums for each of those splits.
            If 23 is chosen, the latitudes are split by:
                90 degN to 23 degN, 23 degN to 23 degS and 23 degS to 90 degS.
            If 30 is chosen, the latitudes are split by:
                90 degN to 30 degN, 30 degN to 30 degS and 30 degS to 90 degS.
            And so on.

            Default is 30.

        lon_range: list-like, optional

            Range of longitudinal values to sum. Other longitudes are ignored.
            Defaults to None, which sums over all longitudinal values.

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

        lat = df.latitude
        lon = df.longitude
        earth_grid_area = earth_area_grid(lat,lon)

        days = {'01': 31, '02': 28, '03': 31, '04': 30,
                '05': 31, '06': 30, '07': 31, '08': 31,
                '09': 30, '10': 31, '11': 30, '12': 31}

        values = {
        "Earth_Land": [], "South_Land": [], "Tropical_Land": [],
        "North_Land": [], "Earth_Ocean": [], "South_Ocean": [],
        "Tropical_Ocean": [], "North_Ocean": []}

        lat_conditions = (
        True, lat < -lat_split,
        (lat>-lat_split) & (lat<lat_split), lat>lat_split
        )


        for index, time_point in enumerate(arg_time_range):

            days_in_month = days[time_point[-2:]]

            earth_land_flux = (
            df['Terrestrial_flux']
                .sel(time=time_point).values[0]*(days_in_month/365)
                )

            # Rayner has ocean variable as 'ocean' instead of 'Ocean_flux'.
            try:
                earth_ocean_flux = (
                df['Ocean_flux']
                    .sel(time=time_point).values[0]*(days_in_month/365)
                    )
            except KeyError:
                earth_ocean_flux = (
                df['ocean']
                    .sel(time=time_point).values[0]*(days_in_month/365)
                    )

            earth_land_sink = earth_grid_area*earth_land_flux
            earth_ocean_sink = earth_grid_area*earth_ocean_flux

            conditions = iter(lat_conditions)
            for var in values[:4]:
                sum = np.sum(1e-15*earth_land_sink)[next(conditions)]
                values[var].append(sum)

            conditions = iter(lat_conditions)
            for var in values[4:]:
                sum = np.sum(1e-15*earth_ocean_sink)[next(conditions)]
                values[var].append(sum)

            time_vals.append(df.sel(time=time_point).time.values[0]))


        ds = xr.Dataset(
            {key: (('time', value) for (key, value) in values},
            coords={'time': (('time'), time_vals)}
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

        time_list = [datetime.strptime(time.strftime('%Y-%m'), '%Y-%m') for time in data.time.values]
        self.time = pd.to_datetime(time_list)



    def rolling_trend(self, variable, window_size=25, plot=False,
    include_pearson=False):
        """ Calculates the slope of the trend of an uptake variable for each
        time window and for a given window size. The function also plots the
        slopes as a timeseries and, if prompted, the r-value of each slope as
        a timeseries.

        Parameters
        ==========

        variable: string

            carbon uptake variable to regress.

        window_size: integer

            size of time window of trends.

        plot: bool, optional

            Option to show plots of the slopes.
            Defaults to False.

        include_pearson: bool, optional

            Option to include dataframe and plot of
            r-values for each year.

            Defaults to False.

        """

        def to_numeric(date):
            return date.year + (date.month-1 + date.day/31)/12

        roll_vals = []
        r_vals = []

        for i in range(0, len(self.time) - window_size):
            sub_time = self.time[i:i+window_size+1]
            sub_vals = self.data[variable].sel(time = slice(self.data.time[i],
            self.data.time[i+window_size])).values

            linreg = stats.linregress(to_numeric(sub_time), sub_vals)

            roll_vals.append(linreg[0])
            r_vals.append(linreg[2])


        roll_df = pd.DataFrame({f"{window_size}-year trend slope": roll_vals},
        index=to_numeric(self.time[:-window_size]))

        if plot:

            plt.figure(figsize=(22,16))

            plt.subplot(211)
            plt.plot(self.time, self.data[variable].values)
            plt.ylabel("C flux to the atmosphere (GtC)", fontsize=20)

            plt.subplot(212)
            plt.plot(roll_df, color='g')
            plt.ylabel("Slope of C flux trend (GtC/ppm/yr)", fontsize=20)

        if include_pearson:
            r_df = pd.DataFrame({"r-values of trends": r_vals},
            index=to_numeric(self.time[:-window_size]))
            return roll_df, r_df
        else:
            return roll_df

    def psd(self, variable, fs=1, xlim=None, plot=False):
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

        if fs == 12 or fs==1: #for analysis.py: fs==12 means that annual
        # resolution timeseries is being passeD as self. fs==1 means
        # resolution is annual.
            period = " (years)"
            unit = "((GtC/yr)$^2$.yr)"
        else:
            period = ""
            unit = ""

        x = self.data[variable]
        freqs, spec = signal.welch(x.values, fs=fs)

        if plot:
            plt.figure(figsize=(12,9))

            plt.subplot(211)
            plt.plot(self.time, x.values)

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
            indices = range(i, len(x)+i, 12)
            sub = x[indices]
            mean_list.append(np.mean(sub))

        s = []
        for i in range(int(len(x)/12)):
            for j in mean_list:
                s.append(j)
        s = np.array(s)

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
                indices = range(i, len(x)+i, 12)
                sub = x[indices]
                mean_list.append(np.mean(sub))

            s = []
            for i in range(int(len(x)/12)):
                for j in mean_list:
                    s.append(j)
            s = np.array(s)

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

    def regress_rolling_trend_to_GCP(self, sink, window_size, plot=False):
        """Calculates linear regression of model rolling gradient to GCP
        rolling gradient and shows a plot of the rolling gradients and scatter
        plot if requested.

        Parameters:
        ===========

        sink: string

            either land or ocean.

        plot: bool, optional

            Plots rolling gradients and scatter plot of GCP and model uptake
            if True.
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
            .rolling_trend(self, model_sink, window_size).values
            .squeeze()
        )

        GCP_roll_df = (
        GCPf
            .Analysis("land sink")
            .rolling_trend(window_size=window_size)
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
