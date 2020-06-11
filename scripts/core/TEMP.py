"""Perform outputs and analyses for temperature anomalies from HadCRUT gridded
data.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np


""" FUNCTIONS """
class SpatialAve:
    """This class takes an instance of the netCDF datasets from the
    data/temp/crudata folder.
    It provides features to prepare the datasets for compatibility for the
    functions in the Analysis class. This includes the averaging of
    temperature on spatial and temporal scales and conversion of cfdatetime time
    objects to datetime.

    Parameters
    ==========

    data: one of xarray.Dataset, xarray.DataArray and nc.file.

    """

    def __init__(self, data):
        """ Initialise an instance of an SpatialAve. """
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


        if 'temperature_anomaly' in _data.variables:
            _var = 'temperature_anomaly'

        elif 'sst' in _data.variables:
            _var = 'sst'

        else:
            raise ValueError("neither 'temperature_anomaly' or 'sst' were detected in this file.")

        self.var = _var

    def spatial_averaging(self, start_time=None, end_time=None,
    lat_split=30, lon_range=None):
        """Returns a xr.Dataset of total global and regional average temperature at
        a specific time point or a range of time points.
        The regions are split into land and ocean and are latitudinally split
        according to passed argument for lat_split.
        A longitudinal range can also be chosen.
        Globally averaged temperatures are also included for each of land and ocean.

        Parameters
        ==========

        start_time: int, optional

            The year of the time to start the averaging. Must be an integer.
            Default is None.

        end_time: int, optional

            The year of the time to end the averaging. Must be an integer. Note
            that the averaging will stop the year before argument.
            Default is None.

        lat_split: integer, optional

            Split the latitudes and output averages for each of those splits.
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
            start_time = df.time.values[0].__str__()[:7]
        if end_time == None:
            end_time = df.time.values[-1].__str__()[:7]

            year, month = int(end_time[:4]), int(end_time[5:])
            if month == 12:
                end_time = f"{year+1}-01"
            elif month < 9:
                end_time = f"{year}-0{month+1}"
            else:
                end_time = f"{year}-{month+1}"

        arg_time_range = (
        pd
            .date_range(start=start_time, end=end_time, freq='M')
            .strftime('%Y-%m')
        )

        lat = df.latitude
        lon = df.longitude

        values = {
        "Earth": [], "South": [], "Tropical": [], "North": []}

        time_vals = []

        lat_conditions = (
        True, lat < -lat_split,
        (lat>-lat_split) & (lat<lat_split), lat>lat_split
        )

        for time_point in arg_time_range:
            temp = df[self.var].sel(time=time_point).values.squeeze()

            condition = iter(lat_conditions)
            for var in list(values.keys()):
                mean = np.nanmean(temp[next(condition)])
                values[var].append(mean)

            time_vals.append(df.sel(time=time_point).time.values[0])

        ds = xr.Dataset(
            {key: (('time'), value) for (key, value) in values.items()},
            coords={'time': (('time'), time_vals)}
        )

        return ds


""" INPUTS """
# fname = "./../../data/temp/crudata/HadSST.3.1.1.0.median.nc"
# fname = "./../../data/temp/crudata/HadCRUT.4.6.0.0.median.nc"
fname = "./../../data/temp/crudata/CRUTEM.4.6.0.0.anomalies.nc"

""" EXECUTION """
df = SpatialAve(fname)
df.data
df.spatial_averaging()
