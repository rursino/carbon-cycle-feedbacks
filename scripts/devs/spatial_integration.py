"""Development/Fixing of the SpatialAgg.spatial_integration() function in
inv_flux.py.
"""


""" IMPORTS """
import xarray as xr
import numpy as np
import pandas as pd
from itertools import *


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

def spatial_integration(df, start_time=None, end_time=None, lat_split=30):

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

def regional_cut(df, lats, lons, start_time=None, end_time=None): # add: sum

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

    df = df.sel(latitude=slice(*lats), longitude=slice(*lons))
    df = df * (earth_area_grid(df.latitude, df.longitude) * 1e-15)
    df = df.sum(axis=(1,2))
    df = df * (np.ones(252) * 30/365)

    df = df.assign_coords(('time', arg_time_range))

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

""" EXECUTION """
df = xr.open_dataset(fname)

res = regional_cut(df, (30,60), (-30,30))
res

spatial_integration(df)
