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

    def spatial_averaging(self, lat_split=30, start_time=None, end_time=None):
        """Returns a xr.Dataset of total global and regional average temperature at
        a specific time point or a range of time points.
        The regions are split into land and ocean and are latitudinally split
        according to passed argument for lat_split.
        A longitudinal range can also be chosen.
        Globally averaged temperatures are also included for each of land and ocean.

        Parameters
        ==========

        lat_split: integer, optional

            Split the latitudes and output averages for each of those splits.
            If 23 is chosen, the latitudes are split by:
                90 degN to 23 degN, 23 degN to 23 degS and 23 degS to 90 degS.
            If 30 is chosen, the latitudes are split by:
                90 degN to 30 degN, 30 degN to 30 degS and 30 degS to 90 degS.
            And so on.

            Default is 30.

        start_time: int, optional

            The (year, month) of the time to start the averaging. Must be an
            integer.
            Default is None.

        end_time: int, optional

            The (year, month) of the time to end the averaging. Must be an
            integer. Note that the averaging will stop the year before argument.
            Default is None.

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

    def time_range(self, start_time, end_time, slice_obj=False):
        """ Returns a list or slice object of a range of time points, as is
        required when selecting time points for other functions.

        Parameters
        ==========

        start_time: string

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            If None is passed, then start_time refers to the first time point
            in the Dataset.

        end_time: string

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            If None is passed, then start_time refers to the last time point
            in the Dataset.

        slice_obj: bool, optional

            If True, then function returns a slice object instead of a list.
            Defaults to False.

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

            The month and year of the time to start the integration in the
            format '%Y-%M'.
            Default is None.

        end_time: string, optional

            The month and year of the time to end the integration in the
            format '%Y-%M'. Note that the integration will stop the month
            before argument.
            Default is None.

        """

        df = self.data[self.var]

        arg_time_range = self.time_range(start_time, end_time)
        slice_time_range = self.time_range(start_time, end_time,
                                           slice_obj=True)

        df = df.sel(latitude=slice(*lats), longitude=slice(*lons),
                    time=slice_time_range)

        df = df.mean(axis=(1,2), skipna=True)

        df['time'] = arg_time_range

        # Degree math symbol
        deg = u'\N{DEGREE SIGN}'
        df = df.assign_attrs(
                            {
                            "latitudes": "{}{} - {}{}".format(lats[1], deg,
                                                              lats[0], deg),
                            "longitudes": "{}{} - {}{}".format(lons[0], deg,
                                                               lons[1], deg)
                            }
                            )

        return df

    def latitudinal_splits(self, lat_split=30, start_time=None, end_time=None):
        """Returns a xr.Dataset of total global and regional average
        temperature at a specific time point or a range of time points.

        The regions are split into land and ocean and are latitudinally split
        according to passed argument for lat_split.
        This function uses the regional_cut function to cut the gridded
        temperature to the latitudes required for the latitudinal splits.

        Globally averaged temperatures are also included for each of land and
        ocean.

        Parameters
        ==========

        lat_split: integer, optional

            Split the latitudes and output averages for each of those splits.
            If 23 is chosen, the latitudes are split by:
                90 degN to 23 degN, 23 degN to 23 degS and 23 degS to 90 degS.
            If 30 is chosen, the latitudes are split by:
                90 degN to 30 degN, 30 degN to 30 degS and 30 degS to 90 degS.
            And so on.

            Default is 30.

        start_time: int, optional

            The (year, month) of the time to start the averaging. Must be an
            integer.
            Default is None.

        end_time: int, optional

            The (year, month) of the time to end the averaging. Must be an
            integer. Note that the averaging will stop the year before argument.
            Default is None.

        """

        vars = {
                "Earth": (-90, 90),
                "South": (-90, -lat_split),
                "Tropical": (-lat_split, lat_split),
                "North": (lat_split, 90)
                }

        if np.all(np.diff(self.data.latitude.values) < 0):
            for var in vars:
                vars[var] = vars[var][::-1]

        values = {}
        for var in vars:
            lat_split = vars[var]
            region_df = self.regional_cut(lat_split, (-180,180),
                                    start_time=start_time, end_time=end_time)
            values[var] = region_df.values

        slice_time_range = self.time_range(start_time, end_time, slice_obj=True)
        ds_time = self.data.sel(time=slice_time_range).time.values

        ds = xr.Dataset(
            {key: (('time'), value) for (key, value) in values.items()},
            coords={'time': (('time'), ds_time)}
        )

        return ds
