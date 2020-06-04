"""Development/Fixing of the SpatialAgg.spatial_integration() function in
inv_flux.py.
"""


""" IMPORTS """

import sys
sys.path.append("./../core/")
import inv_flux

import xarray as xr
import numpy as np
import pandas as pd


""" INPUTS """

dir = "./../../data/inversions/"
file = "fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc"

fname = dir + file


""" FUNCTIONS"""

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

        time_vals = []

        lat_conditions = (
        True, lat < -lat_split,
        (lat>-lat_split) & (lat<lat_split), lat>lat_split
        )


        for time_point in arg_time_range:

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

            condition = iter(lat_conditions)
            for var in list(values.keys())[:4]:
                sum = np.sum(1e-15*earth_land_sink[next(condition)])
                values[var].append(sum)

            condition = iter(lat_conditions)
            for var in list(values.keys())[4:]:
                sum = np.sum(1e-15*earth_ocean_sink[next(condition)])
                values[var].append(sum)

            time_vals.append(df.sel(time=time_point).time.values[0])

        ds = xr.Dataset(
            {key: (('time'), value) for (key, value) in values.items()},
            coords={'time': (('time'), time_vals)}
        )

        return ds


""" EXECUTION """
df = SpatialAgg(data=fname)
result = df.spatial_integration()

result

""" TEST """
import pickle

ftest = "./../../output/inversions/raw/output_all/Rayner_all/spatial.pik"
pickle.load(open(ftest, "rb"))
