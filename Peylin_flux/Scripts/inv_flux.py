import numpy as np
import xarray as xr
import sys
import pandas as pd




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
    It provides features to analyse the carbon flux data from the netCDF files,
    including the integration of fluxes on spatial and temporal scales.
    
    Parameters
    ----------
    data: an instance of an xarray.Dataset, xarray.DataArray or .nc file.
    
    """

    def __init__(self, data):
        """ Initialise an instance of an TheDataFrame. """
        if isinstance(data, xr.Dataset) or isinstance(data, xr.DataArray):
            _data = data
        else:
            _data = xr.open_dataset(data)
        
        self.data = _data


    def spatial_integration(self, start_time=None, end_time=None):
        """Returns a pd.DataFrame of total global and regional sinks at a specific time point or a range of time points.
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

        columns = ['time',
                   'earth_land_total', 'south_land_total', 'trop_land_total', 'north_land_total',
                   'earth_ocean_total','south_ocean_total', 'trop_ocean_total','north_ocean_total']
        total_values = pd.DataFrame(columns=columns)

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

            total_values.loc[index,:] = np.array([df.sel(time=time_point).time.values[0],
                                              np.sum(1e-15*earth_land_sink),
                                              np.sum(1e-15*earth_land_sink[lat<-23]),
                                              np.sum(1e-15*earth_land_sink[(lat>-23) & (lat<23)]),
                                              np.sum(1e-15*earth_land_sink[lat>23]),
                                              np.sum(1e-15*earth_ocean_sink),
                                              np.sum(1e-15*earth_ocean_sink[lat<-23]),
                                              np.sum(1e-15*earth_ocean_sink[(lat>-23) & (lat<23)]),
                                              np.sum(1e-15*earth_ocean_sink[lat>23])])


        return total_values.reset_index(drop=True)


    def year_integration(self):
        """ Returns a pd.DataFrame with yearly integrated regional fluxes. """
        
        df = self.spatial_integration()

        min_year = df.time[0].year
        max_year = df.time[df.time.size-1].year

        df_year = pd.DataFrame(columns= 
                               ['Year',
                                'earth_land_total', 'south_land_total', 'trop_land_total', 'north_land_total',
                                'earth_ocean_total','south_ocean_total', 'trop_ocean_total','north_ocean_total']
                              )

        for (j,year) in enumerate(range(min_year, max_year+1)):
            index = []
            for (i,time) in enumerate(df.time):
                if time.year == year:
                    index.append(i)
            df_year.loc[j,:] = df.iloc[index,1:].sum()

        df_year['Year'] = range(min_year, max_year+1)
        
        return df_year


    def decade_integration(self):
        """ Returns a pd.DataFrame with decadal integrated regional fluxes. """
        
        df = self.spatial_integration()

        min_decade = int(df.time[0].year/10)*10
        max_decade = int(df.time[df.time.size-1].year/10)*10

        df_decade = pd.DataFrame(columns= 
                               ['Decade',
                                'earth_land_total', 'south_land_total', 'trop_land_total', 'north_land_total',
                                'earth_ocean_total','south_ocean_total', 'trop_ocean_total','north_ocean_total']
                              )

        for (j,decade) in enumerate(range(min_decade, max_decade+10,10)):
            index = []
            for (i,time) in enumerate(df['time']):
                if time.year in range(decade, decade+10):
                    index.append(i)
            df_decade.loc[j,:] = df.iloc[index,1:].sum()

        df_decade['Decade'] = range(min_decade, max_decade+10,10)
        return df_decade


    def whole_time_integration(self):
        """ Returns a pd.DataFrame with regional fluxes integrated through the entire time period. """
        
        df = self.spatial_integration()
        
        df_whole_time = pd.DataFrame(columns=
                                     ['earth_land_total', 'south_land_total', 'trop_land_total', 'north_land_total',
                                      'earth_ocean_total','south_ocean_total', 'trop_ocean_total','north_ocean_total']
                                    )
        df_whole_time.loc[0,:] = df.iloc[:,1:].sum()
        
        return df_whole_time
    
    def single_time_output(self, fname, time):
        """ Writes a text file of spatially integrated regional fluxes at a specific time point.

        Parameters
        ----------
        fname: Name of the file to be written.
        time: Must be a single date string in the format %Y-%M.
        
        """
        
        df = self.spatial_integration()
        
        time_year = int(time[:4])
        time_month = int(time[-2:])
        time_indices = np.array([(datetime.year, datetime.month) for i,datetime in enumerate(df.time.values)])
        year_index = np.where(time_year == time_indices[:,0])
        month_index = np.where(time_month == time_indices[year_index,1][0])[0][0]
        time_index = year_index[0][month_index]

        output = open(fname, "w")
        output.write(
            
            "Time: " + str(df.loc[time_index,'time']) + "\n\n"
            "Earth Land Total Sink: " + str(df.loc[time_index,'earth_land_total']) +" GtC\n"
            "South Land Total Sink: " + str(df.loc[time_index,'south_land_total']) +" GtC\n"
            "Tropical Land Total Sink: " + str(df.loc[time_index,'trop_land_total']) +" GtC\n"
            "North Land Total Sink: " + str(df.loc[time_index,'north_land_total']) +" GtC\n\n"
            "Earth Ocean Total Sink: " + str(df.loc[time_index,'earth_ocean_total']) +" GtC\n"
            "South Ocean Total Sink: " + str(df.loc[time_index,'south_ocean_total']) +" GtC\n"
            "Tropical Ocean Total Sink: " + str(df.loc[time_index,'trop_ocean_total']) +" GtC\n"
            "North Ocean Total Sink: " + str(df.loc[time_index,'north_ocean_total']) +" GtC\n"
        )
        output.close()

