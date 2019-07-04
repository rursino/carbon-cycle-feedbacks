import numpy as np
import xarray as xr
import sys

# Common values.
from numpy import pi
r_earth = 6.371e6 # Radius of Earth
dtor = pi/180. # conversion from degrees to radians.


def scalar_earth_area(minlat, maxlat, minlon, maxlon):
    
    "Returns the area of earth in the defined grid box."
    
    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)
    
    return 2.*pi*r_earth**2 *(np.sin( dtor*maxlat) - np.sin(dtor*minlat))*(diff_lon[1]-diff_lon[0])/360.


def earth_area(minlat, maxlat, minlon, maxlon):
    
    "Returns grid of areas with shape=(minlon, minlat) and earth area."
    
    result = np.zeros((np.array(minlon).size, np.array(minlat).size))
    diffsinlat = 2.*pi*r_earth**2 *(np.sin(dtor*maxlat) - np.sin(dtor*minlat))
    diff_lon = np.unwrap((minlon,maxlon), discont=360.+1e-6)
    
    for i in range(np.array(minlon).size):
        result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.
    
    return result

def earth_area_grid(lats,lons):
    
    "Returns an array of the areas of each grid box within a defined set of lats and lons."
    
    result=np.zeros((lats.size, lons.size))
    
    minlats = lats - 0.5*(lats[1]-lats[0])
    maxlats= lats + 0.5*(lats[1]-lats[0])
    minlons = lons - 0.5*(lons[1]-lons[0])
    maxlons=lons + 0.5*(lons[1]-lons[0])
    
    for i in range(lats.size):
        result[i,:] = scalar_earth_area(minlats[i],maxlats[i],minlons[0],maxlons[0])
    
    return result

# Code to get aggregate fluxes.

# Open Rayner's netCDF file.
Rayner = './../data/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc'
CAMS_17 = './../data/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc'

data = Rayner
df = xr.open_dataset(data)

# Obtain object names for variables
if 'latitude' and 'longitude' and 'time' in df.variables:
    lat = df.latitude
    lon = df.longitude
    time = df.time
    
# Obtain grid areas of globe and regions.
earth_grid_area = earth_area_grid(lat,lon)
south_grid_area = earth_area_grid(lat[lat<-23], lon)
trop_grid_area = earth_area_grid(lat[(lat>-23) & (lat<23)], lon)
north_grid_area = earth_area_grid(lat[lat>23], lon)

# Obtain grid values of fluxes.
time_point = str(time.values[0])[:10]

earth_land_flux = df.Terrestrial_flux.sel(time=time_point).values[0]
south_land_flux = earth_land_flux[lat<-23]
trop_land_flux = earth_land_flux[(lat>-23) & (lat<23)]
north_land_flux = earth_land_flux[lat>23]

earth_ocean_flux = df.ocean.sel(time=str(time.values[0])[:10]).values[0]
south_ocean_flux = earth_ocean_flux[lat<-23]
trop_ocean_flux = earth_ocean_flux[(lat>-23) & (lat<23)]
north_ocean_flux = earth_ocean_flux[lat>23]

# Obtain grid values of total sink.
earth_land_sink = earth_grid_area*earth_land_flux
south_land_sink = south_grid_area*south_land_flux
trop_land_sink = trop_grid_area*trop_land_flux
north_land_sink = north_grid_area*north_land_flux

earth_ocean_sink = earth_grid_area*earth_ocean_flux
south_ocean_sink = south_grid_area*south_ocean_flux
trop_ocean_sink = trop_grid_area*trop_ocean_flux
north_ocean_sink = north_grid_area*north_ocean_flux

# Obtain total values of sinks in globe and regions.
earth_land_total = np.sum(earth_land_sink)*1e-15
south_land_total = np.sum(south_land_sink)*1e-15
trop_land_total = np.sum(trop_land_sink)*1e-15
north_land_total = np.sum(north_land_sink)*1e-15

earth_ocean_total = np.sum(earth_ocean_sink)*1e-15
south_ocean_total = np.sum(south_ocean_sink)*1e-15
trop_ocean_total = np.sum(trop_ocean_sink)*1e-15
north_ocean_total = np.sum(north_ocean_sink)*1e-15

# Output of results.
output = open("output.txt", "w")
output.write(
    "netCDF file path: " + data +"\n"
    "Time: " + time_point + "\n\n\n"
    "Earth Land Total Sink: " + str(earth_land_total) +" GtC\n"
    "South Land Total Sink: " + str(south_land_total) +" GtC\n"
    "Tropical Land Total Sink: " + str(trop_land_total) +" GtC\n"
    "North Land Total Sink: " + str(north_land_total) +" GtC\n\n"
    "Earth Ocean Total Sink: " + str(earth_ocean_total) +" GtC\n"
    "South Ocean Total Sink: " + str(south_ocean_total) +" GtC\n"
    "Tropical Ocean Total Sink: " + str(trop_ocean_total) +" GtC\n"
    "North Ocean Total Sink: " + str(north_ocean_total) +" GtC\n"
)
output.close()
