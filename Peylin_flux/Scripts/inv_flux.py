import numpy as np
import xarray as xr
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats



""" The following three functions obtain the area of specific grid boxes of the Earth in different formats.
It is used in the spatial_integration function within the TheDataFrame class. """

from numpy import pi
r_earth = 6.371e6 # Radius of Earth
dtor = pi/180. # conversion from degrees to radians.

def scalar_earth_area(minlat, maxlat, minlon, maxlon):
    """Returns the area of earth in the defined grid box."""
    
    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)
    
    return 2.*pi*r_earth**2 *(np.sin( dtor*maxlat) - np.sin(dtor*minlat))*(diff_lon[1]-diff_lon[0])/360.


def earth_area(minlat, maxlat, minlon, maxlon):
    """Returns grid of areas with shape=(minlon, minlat) and earth area."""
    
    result = np.zeros((np.array(minlon).size, np.array(minlat).size))
    diffsinlat = 2.*pi*r_earth**2 *(np.sin(dtor*maxlat) - np.sin(dtor*minlat))
    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)
    
    for i in range(np.array(minlon).size):
        result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.
    
    return result

def earth_area_grid(lats,lons):
    """Returns an array of the areas of each grid box within a defined set of lats and lons."""
    
    result=np.zeros((lats.size, lons.size))
    
    minlats = lats - 0.5*(lats[1]-lats[0])
    maxlats= lats + 0.5*(lats[1]-lats[0])
    minlons = lons - 0.5*(lons[1]-lons[0])
    maxlons=lons + 0.5*(lons[1]-lons[0])
    
    for i in range(lats.size):
        result[i,:] = scalar_earth_area(minlats[i],maxlats[i],minlons[0],maxlons[0])
    
    return result




class TheDataFrame:
    """ This class takes an instance of the netCDF datasets from the Peylin_flux/data folder.
    It provides features to prepare the datasets for compatibility for the functions in the inv.Analysis class. This includes the integration of fluxes on spatial and temporal scales and conversion of cfdatetime time objects to datetime.
    
    Parameters
    ----------
    data: an instance of an xarray.Dataset, xarray.DataArray or .nc file.
    
    """

    def __init__(self, data):
        """ Initialise an instance of an TheDataFrame. """
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

    
    def spatial_integration(self, start_time=None, end_time=None):
        """Returns a xr.Dataset of total global and regional sinks at a specific time point or a range of time points.
        The regions are split into land and ocean and are latitudinally split as follows:
        90 degN to 23 degN, 23 degN to 23 degS and 23 degS to 90 degS.
        Globally integrated fluxes are also included for each of land and ocean.
        
        Parameters
        ----------
        start_time: string, default None
            The month and year of the time to start the integration in the format '%Y-%M'.
            
        end_time: string, default None
            The month and year of the time to end the integration in the format '%Y-%M'.
            Note that the integration will stop the month before argument.
        
        """
        
        df = self.data

        values = {'Time':[],
                  'Earth_Land':[], 'South_Land':[], 'Tropical_Land':[], 'North_Land':[],
                  'Earth_Ocean':[],'South_Ocean':[], 'Tropical_Ocean':[],'North_Ocean':[]}

        if start_time == None:
            start_time = df.time.values[0].strftime('%Y-%m')
        if end_time == None:
            end_time = df.time.values[-1]
            try:
                next_month = end_time.replace(month=end_time.month+1)
            except ValueError:
                next_month = end_time.replace(year=end_time.year+1, month=1)

            end_time = next_month.strftime('%Y-%m')
        
        arg_time_range = pd.date_range(start=start_time, end=end_time, freq='M').strftime('%Y-%m')

        lat = df.latitude
        lon = df.longitude
        earth_grid_area = earth_area_grid(lat,lon)

        days = {'01': 31, '02': 28, '03': 31, '04': 30,
                '05': 31, '06': 30, '07': 31, '08': 31,
                '09': 30, '10': 31, '11': 30, '12': 31}

        for index,time_point in enumerate(arg_time_range):
            
            days_in_month = days[time_point[-2:]]

            earth_land_flux = df['Terrestrial_flux'].sel(time=time_point).values[0]*(days_in_month/365)
            
            # Rayner dataset has ocean variable as 'ocean' instead of 'Ocean_flux'.
            try:
                earth_ocean_flux = df['Ocean_flux'].sel(time=time_point).values[0]*(days_in_month/365)
            except KeyError:
                earth_ocean_flux = df['ocean'].sel(time=time_point).values[0]*(days_in_month/365)

            earth_land_sink = earth_grid_area*earth_land_flux
            earth_ocean_sink = earth_grid_area*earth_ocean_flux

            values['Time'].append(df.sel(time=time_point).time.values[0])
            values['Earth_Land'].append(np.sum(1e-15*earth_land_sink))
            values['South_Land'].append(np.sum(1e-15*earth_land_sink[lat<-23]))
            values['Tropical_Land'].append(np.sum(1e-15*earth_land_sink[(lat>-23) & (lat<23)]))
            values['North_Land'].append(np.sum(1e-15*earth_land_sink[lat>23]))
            values['Earth_Ocean'].append(np.sum(1e-15*earth_ocean_sink))
            values['South_Ocean'].append(np.sum(1e-15*earth_ocean_sink[lat<-23]))
            values['Tropical_Ocean'].append(np.sum(1e-15*earth_ocean_sink[(lat>-23) & (lat<23)]))
            values['North_Ocean'].append(np.sum(1e-15*earth_ocean_sink[lat>23]))
            
        
        ds = xr.Dataset({'Earth_Land': (('time'), values['Earth_Land']),
                         'South_Land': (('time'), values['South_Land']),
                         'Tropical_Land': (('time'), values['Tropical_Land']),
                         'North_Land': (('time'), values['North_Land']),
                         'Earth_Ocean': (('time'), values['Earth_Ocean']),
                         'South_Ocean': (('time'), values['South_Ocean']),
                         'Tropical_Ocean': (('time'), values['Tropical_Ocean']),
                         'North_Ocean': (('time'), values['North_Ocean'])},
                        coords={'time': (('time'), values['Time'])}
          )
        
        return ds
    
    def cftime_to_datetime(self, format='%Y-%m'):
        """Takes a xr.Dataset with cftime values and converts them into datetimes.

        Parameters
        ----------
        self: xr.Dataset.
        format: format of datetime.
        ----------
        """

        time_list = []
        for time in self.data.time.values:
            time_value = datetime.strptime(time.strftime('%Y-%m'), '%Y-%m')
            time_list.append(time_value)

        return self.data.assign_coords(time=time_list)


class Analysis:
    """ This class takes an instance of spatially and/or temporally integrated datasets and provides
    analysis and visualisations on the data as required, including linear regression over whole and
    decadal periods and ...
    
    Parameters
    ----------
    data: an instance of an xarray.Dataset.
    
    """

    def __init__(self, data):
        self.data = data
    
    
    def cftime_to_datetime(self, format='%Y-%m'):
        """Takes a xr.Dataset with cftime values and converts them into datetimes.

        Parameters
        ----------
        self: xr.Dataset.
        format: format of datetime.
        
        """

        time_list = []
        for time in self.data.time.values:
            time_value = datetime.strptime(time.strftime(format), format)
            time_list.append(time_value)

        return self.data.assign_coords(time=time_list)
    
    
    def linear_regression_time(self, time, variable, save_plot=False):
        """ Displays a plot of variable chosen from dataset with coordinate time or CO2 and its linear regression over the whole period.
        
        Parameters
        ----------
        time: dictionary of np.array objects of datetime values.
        variable: variable to regress.
        save_plot: save the plot as a jpg file.
        
        """
        
        def time_list(t):
            x = []
            for year in t:
                x.append(int(year.astype('str')[:4]))
            return np.array(x)
        
        years=time_list(self.data['time'].values)
        plt.plot(years, self.data[variable])
        plt.title(f'{variable} Uptake with decadal trends')
        plt.xlabel('Year')
        plt.ylabel('C flux to the atmosphere (GtC/yr)')
        plt.xticks(np.arange(years[0], years[-1], 5))
        
        regression_list = {}
        for period in time.keys():
            x = time_list(time[period])
            y=self.data[variable].sel({'time': time[period]}).values

            regression = stats.linregress(x, y)
            slope = regression[0]
            intercept = regression[1]
            regression_list[period] = regression

            line = slope*x+intercept

            plt.plot(x, line)
        
        if type(save_plot)==str:
            plt.savefig(save_plot)
        
        return regression_list
