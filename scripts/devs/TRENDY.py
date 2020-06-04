"""Development of the TRENDY_flux.py script.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np


""" INPUTS """
fname = "./../../data/TRENDY/models/LPJ-GUESS/S0/LPJ-GUESS_S0_cVeg.nc"


""" FUNCTIONS """
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
    earth_grid_area = self.earth_area_grid(lat,lon)

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
df = xr.open_dataset(fname)
spatial_integration(df)
