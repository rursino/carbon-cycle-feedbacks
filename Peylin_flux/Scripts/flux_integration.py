import numpy as np
import xarray as xr
import sys
import earth_area as ea

def integrate_spatial(data, minlat, maxlat):
    
    ''' Insert '''
    
    df = xr.open_dataset(data)
    lat = df.latitude
    lon = df.longitude
    
    grid = np.zeros((lat.size, lon.size)) # Initialise a dataframe for the 