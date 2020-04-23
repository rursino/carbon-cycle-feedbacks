import numpy as np
import xarray as xr
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats, signal



class Analysis:
    """ This class takes the budget.csv in the GCP data folder and provides analysis and visualisations
    on selected data as required, including plotting timeseries, linear regression over whole an decadal
    periods and frequency analyses.
    
    Parameters
    ----------
    variable: the name of a selected column from budget.csv.
    
    """

    # START EDITING HERE (written 23/4)
    def __init__(self, data):
        """Take the xr.Dataset with cftime values and converts them into datetimes."""        
        
        self.data = data
        
        time_list = [datetime.strptime(time.strftime('%Y-%m'), '%Y-%m') for time in data.time.values]
        self.time = pd.to_datetime(time_list)
        
    
       
    def rolling_trend(self, variable, window_size=25, plot=False, include_pearson=False):
        """ Calculates the slope of the trend of an uptake variable for each time window and for a given window size. The function also plots the slopes as a timeseries and, if prompted, the r-value of each slope as a timeseries.

        Parameters
        ----------
        variable: carbon uptake variable to regress.
        window_size: size of time window of trends.
        plot: Defaults to False. Option to show plots of the slopes.
        include_pearson: Defaults to False. Option to include dataframe and plot of r-values for each year.

        """
        
        def to_numeric(date):
            return date.year + (date.month-1 + date.day/31)/12

        roll_vals = []
        r_vals = []
        
        for i in range(0, len(self.time) - window_size):
            sub_time = self.time[i:i+window_size+1]
            sub_vals = self.data[variable].sel(time = slice(self.data.time[i], self.data.time[i+window_size])).values
            
            linreg = stats.linregress(to_numeric(sub_time), sub_vals)
            
            roll_vals.append(linreg[0])
            r_vals.append(linreg[2])
        
        
        roll_df = pd.DataFrame({f"{window_size}-year trend slope": roll_vals}, index=to_numeric(self.time[:-window_size]))
        
        if plot:
        
            plt.figure(figsize=(22,16))
            
            plt.subplot(211)
            plt.plot(self.time, self.data[variable].values)
            plt.ylabel("C flux to the atmosphere (GtC)", fontsize=20)

            plt.subplot(212)
            plt.plot(roll_df, color='g')
            plt.ylabel("Slope of C flux trend (GtC/ppm/yr)", fontsize=20)           

        if include_pearson:
            r_df = pd.DataFrame({"r-values of trends": r_vals}, index=to_numeric(self.time[:-window_size]))
            return roll_df, r_df
        else:
            return roll_df


    def psd(self, variable, fs=1, xlim=None, plot=False):
        """ Calculates the power spectral density (psd) of a timeseries of a variable using the Welch method. Also provides the timeseries plot and psd plot if passed. This function is designed for the monthly timeseries, but otherwise plotting the dataframe is possible.
        
        Parameters
        ----------
        variable: carbon uptake variable to calculate psd.
        fs: sampling frequency.
        xlim: apply limit to x-axis of the psd. Must be a list of two values.
        plot: defaults to False. If assigned to True, shows two plots of timeseries and psd.
        
        """

        if fs == 12 or fs==1: #for analysis.py: fs==12 means that annual resolution timeseries is being passed as self. fs==1 means resolution is annual.
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

        return pd.DataFrame({f"Period{period}": 1/freqs, f"Spectral Variance {unit}": spec}, index=freqs)
    
    
    def deseasonalise(self, variable):
        """ Deseasonalise a timeseries of a variable by applying and using a seasonal index.
        
        Parameters:
        -----------
        variable: variable to apply from self object.
        
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
    
    
    def bandpass(self, variable, fc, fs=1, order=5, btype="low", deseasonalise_first=False):
        """ Applies a bandpass filter to a dataset (either lowpass, highpass or bandpass) using the scipy.signal.butter function.

        Parameters:
        -----------
        variable: variable to apply from self object.
        fc: cut-off frequency or frequencies.
        fs: sample frequency of x.
        order: order of the filter. Defaults to 5.
        btype: options are low, high and band.
        deseasonalise_first: Defaults to False. Option to deseasonalise timeseries before applying bandpass filter.

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
    """ This class takes an instance of the model uptake datasets and provides evaluations
    against the Global Carbon Project (GCP) uptake timeseries. Must be annual resolution to match GCP.
    
    Parameters
    ----------
    data: an instance of an xarray.Dataset.
    
    """

    
    def __init__(self, data):
        """Take the xr.Dataset with cftime values and converts them into datetimes."""

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
        
        GCP['CO2'] = pd.read_csv("./../../data/CO2/co2_global.csv", index_col=0, header=0)[2:]
        GCP['land sink'] = -GCP['land sink']
        GCP['ocean sink'] = -GCP['ocean sink']
        GCP['budget imbalance'] = -GCP["budget imbalance"] + GCP['land sink']
        GCP.rename(columns={"ocean sink": "ocean",
                            "land sink": "land (model)",
                            "budget imbalance": "land"
                           },
                   inplace=True)
        
        self.GCP = GCP
        
        
    
    def plot_vs_GCP(self, sink, x="time"):
        """Plots variable chosen from model uptake timeseries and GCP uptake
        timeseries, either against time or CO2 concentration.
        
        Parameters:
        -----------
        sink: either land or ocean.
        x: x axis; either time or CO2. Defaults to time.
        
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
        """Calculates linear regression of model uptake to GCP uptake and shows a plot
        of the timeseries and scatter plot if requested.
        
        Parameters:
        -----------
        sink: either land or ocean.
        plot: Plots timeseries and scatter plot of GCP and model uptake if True. Defaults to False.
        
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
    
    
    # NOT finished.
    def regress_rolling_trend_to_GCP(self, sink, window_size, plot=False):
        """Calculates linear regression of model rolling gradient to GCP rolling gradient
        and shows a plot of the rolling gradients and scatter plot if requested.
        
        Parameters:
        -----------
        sink: either land or ocean.
        plot: Plots rolling gradients and scatter plot of GCP and model uptake if True. Defaults to False.
        
        """
        
        raise NotImplementedError
        
        def to_numeric(date):
            return date.year + (date.month-1 + date.day/31)/12
        

        if "land" in sink:
            model_sink = "Earth_Land"
        elif "ocean" in sink:
            model_sink = "Earth_Ocean"

        df = self.data
        GCP = self.GCP
        
        model_roll = Analysis.rolling_trend(self, model_sink, window_size).values.squeeze()
        
        GCP_roll = # GCP scripts
        
        # Plot
        if plot:
            plt.figure(figsize=(14,9))
            plt.subplot(211).plot(GCP.index[:-window_size], GCP_roll)
            plt.subplot(211).plot(GCP.index[:-window_size], model_roll)
            plt.legend(["GCP", "model"])
            plt.subplot(212).scatter(GCP_roll, model_roll)

        return stats.linregress(GCP_roll, model_roll)
        
    
    def compare_trend_to_GCP(self, sink, print_results=False):
        """Calculates long-term trend of model uptake (over the whole time range) and GCP uptake.
        Also calculates the percentage difference of the trends. 
        
        Parameters:
        -----------
        sink: either land or ocean.
        
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
        """Plots autocorrelation of model uptake timeseries using pandas.plotting.
                
        Parameters:
        -----------
        variable: variable from self.data.
        """
        
        return pd.plotting.autocorrelation_plot(self.data[variable].values)
