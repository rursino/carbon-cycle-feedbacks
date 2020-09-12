SPATIAL_DIRECTORY = './../../../data/TRENDY/models/'
fVISITS1 = 'VISIT/S1/VISIT_S1_nbp_decode.nc'
fVISITS3 = 'VISIT/S3/VISIT_S3_nbp_decode.nc'
fJSBACH = 'JSBACH/S1/JSBACH_S1_nbp.nc'

import xarray as xr

VISITS1 = xr.open_dataset(SPATIAL_DIRECTORY + fVISITS1, decode_times=False)
VISITS3 = xr.open_dataset(SPATIAL_DIRECTORY + fVISITS3, decode_times=False)
JSBACH = xr.open_dataset(SPATIAL_DIRECTORY + fJSBACH)

JSBACH_time = JSBACH.sel(time=slice('1860', '2018')).time.values

new_VISITS1 = xr.Dataset(
    {'nbp': (('time', 'latitude', 'longitude'), VISITS1.nbp.values)},
    coords={
            'longitude': (('longitude'), VISITS1.lon.values),
            'latitude': (('latitude'), VISITS1.lat.values),
            'time': (('time'), JSBACH_time)
           }
)

new_VISITS3 = xr.Dataset(
    {'nbp': (('time', 'latitude', 'longitude'), VISITS3.nbp.values)},
    coords={
            'longitude': (('longitude'), VISITS3.lon.values),
            'latitude': (('latitude'), VISITS3.lat.values),
            'time': (('time'), JSBACH_time)
           }
)

new_VISITS1.to_netcdf(SPATIAL_DIRECTORY + 'VISIT/S1/VISIT_S1_nbp.nc')
new_VISITS3.to_netcdf(SPATIAL_DIRECTORY + 'VISIT/S3/VISIT_S3_nbp.nc')


xr.open_dataset(SPATIAL_DIRECTORY + 'VISIT/S1/VISIT_S1_nbp.nc')

xr.open_dataset(SPATIAL_DIRECTORY + 'VISIT/S3/VISIT_S3_nbp.nc')



new_VISITS1
new_VISITS3
JSBACH
