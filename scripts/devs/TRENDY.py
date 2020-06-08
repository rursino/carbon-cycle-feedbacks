"""Development of the TRENDY_flux.py script.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np


""" INPUTS """
fname = "./../../data/TRENDY/models/LPJ-GUESS/S0/LPJ-GUESS_S0_cVeg.nc"


""" FUNCTIONS """
def scalar_earth_area(df, minlat, maxlat, minlon, maxlon):
    """Returns the area of earth in the defined grid box."""

    r_earth = 6.371e6 # Radius of Earth
    dtor = np.pi / 180. # conversion from degrees to radians.

    diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

    return 2. * np.pi * r_earth**2 * (np.sin(dtor * maxlat) -
    np.sin(dtor * minlat)) * (diff_lon[1] - diff_lon[0]) / 360.


def earth_area(df, minlat, maxlat, minlon, maxlon):
    """Returns grid of areas with shape=(minlon, minlat) and earth area.
    """

    r_earth = 6.371e6 # Radius of Earth
    dtor = np.pi / 180. # conversion from degrees to radians.

    result = np.zeros((np.array(minlon).size, np.array(minlat).size))
    diffsinlat = 2. * np.pi * r_earth**2 *(np.sin(dtor*maxlat) -
    np.sin(dtor * minlat))
    diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

    for i in range(np.array(minlon).size):
        result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.

    return result

def earth_area_grid(df, lats, lons):
    """Returns an array of the areas of each grid box within a defined set of
    lats and lons.
    """

    result = np.zeros((lats.size, lons.size))

    minlats = lats - 0.5*(lats[1] - lats[0])
    maxlats= lats + 0.5*(lats[1] - lats[0])
    minlons = lons - 0.5*(lons[1] - lons[0])
    maxlons= lons + 0.5*(lons[1] - lons[0])

    for i in range(lats.size):
        result[i,:] = scalar_earth_area(df,
        minlats[i],
        maxlats[i],
        minlons[0],
        maxlons[0]
        )

    return result

def spatial_integration(df, start_time=None, end_time=None,
lat_split=30, lon_range=None):
    """Returns a xr.Dataset of total global and regional sinks at a
    specific time point or a range of time points.
    The regions are split into land and ocean and are latitudinally split
    according to passed argument for lat_split.
    A longitudinal range can also be chosen.
    Globally integrated fluxes are also included for each of land and ocean.

    Parameters
    ==========

    start_time: int, optional

        The year of the time to start the integration. Must be an integer.
        Default is None.

    end_time: int, optional

        The year of the time to end the integration. Must be an integer. Note
        that the integration will stop the year before argument.
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

    if start_time == None:
        start_time = df.time.values[0].strftime('%Y')

    if end_time == None:
        end_time = df.time.values[-1]
        end_time = (
                    end_time
                        .replace(year=end_time.year+1)
                        .strftime('%Y')
                    )

    arg_time_range = (
    pd
        .date_range(start=start_time, end=end_time, freq='Y')
        .strftime('%Y')
    )

    lat = df.latitude
    lon = df.longitude

    earth_grid_area = earth_area_grid(df, lat, lon)

    values = {
    "Earth": [], "South": [], "Tropical": [], "North": []}

    time_vals = []

    lat_conditions = (
    True, lat < -lat_split,
    (lat>-lat_split) & (lat<lat_split), lat>lat_split
    )

    for time_point in arg_time_range:
        flux = df['cVeg'].sel(time=time_point).values[0]
        sink = earth_grid_area * flux

        condition = iter(lat_conditions)
        for var in list(values.keys())[:4]:
            sum = np.sum(1e-15 * sink[next(condition)]) #fix nan
            values[var].append(sum)

        time_vals.append(df.sel(time=time_point).time.values[0])

    ds = xr.Dataset(
        {key: (('time'), value) for (key, value) in values.items()},
        coords={'time': (('time'), time_vals)}
    )

    return ds


""" EXECUTION """
df = xr.open_dataset(fname)
res = spatial_integration(df)
res
