""" Devs on fixing latlon grids of TRENDY input datasets.
"""

""" IMPORTS """
import xarray as xr
import numpy as np
import os

import sys
sys.path.append('./../../core')
import TRENDY_flux as TRENDYf


""" INPUTS """
PATH_DIRECTORY = './../../../data/TRENDY/models/'
models = ('CABLE-POP', 'CLASS-CTEM', 'JSBACH', 'LPJ-GUESS', 'OCN')

earth_radius = 6.371e6

""" FUNCTIONS """
def scalar_earth_area(minlat, maxlat, minlon, maxlon):
    """Returns the area of earth in the defined grid box."""

    dtor = np.pi / 180. # conversion from degrees to radians.

    diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

    return 2. * np.pi * earth_radius**2 * (np.sin(dtor * maxlat) -
    np.sin(dtor * minlat)) * (diff_lon[1] - diff_lon[0]) / 360.


def earth_area(minlat, maxlat, minlon, maxlon):
    """Returns grid of areas with shape=(minlon, minlat) and earth area.
    """

    dtor = np.pi / 180. # conversion from degrees to radians.

    result = np.zeros((np.array(minlon).size, np.array(minlat).size))
    diffsinlat = 2. * np.pi * earth_radius**2 *(np.sin(dtor*maxlat) -
    np.sin(dtor * minlat))
    diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

    for i in range(np.array(minlon).size):
        result[i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.

    return result

def earth_area_grid(lats, lons):
    """Returns an array of the areas of each grid box within a defined set of
    lats and lons.
    """

    result = np.zeros((lats.size, lons.size))

    minlats = lats - 0.5*(lats[1] - lats[0])
    maxlats= lats + 0.5*(lats[1] - lats[0])
    minlons = lons - 0.5*(lons[1] - lons[0])
    maxlons= lons + 0.5*(lons[1] - lons[0])

    for i in range(lats.size):
        result[i,:] = scalar_earth_area(
        minlats[i],
        maxlats[i],
        minlons[0],
        maxlons[0]
        )

    return result


""" EXECUTION """
# lat-lons for TRENDY.
TRENDY_latlons = {}
for model in models:
    fname = f'{PATH_DIRECTORY}{model}/S1/{model}_S1_nbp.nc'
    df = xr.open_dataset(fname)

    lats = df.latitude.values
    lons = df.longitude.values
    TRENDY_latlons[model] = ((lats[0], lats[-1], lats[1] - lats[0]),
                             (lons[0], lons[-1], lons[1] - lons[0]))
    # TRENDY_latlons[model] = lats, lons

TRENDY_latlons['LPJ-GUESS'] # Good
TRENDY_latlons['OCN'] # Good
TRENDY_latlons['CLASS-CTEM'] # Regrid
TRENDY_latlons['CABLE-POP'] # Shift latitude vals 4deg South (Regrid may be a
                            # shortcut solution)
TRENDY_latlons['JSBACH'] # Regrid

# lat-lons for inversions.
inv_fname = ('./../../../data/inversions/'
            'fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc')

df = xr.open_dataset(fname)
lats = df.latitude.values
lons = df.longitude.values
inv_latlons = lats, lons
inv_latlons
# inversion lat-lon grid is the same as OCN.


# Calculate surface area of TRENDY lat-lon grids.
earth_area_grid(*TRENDY_latlons['CABLE-POP']).sum() - earth_radius**2 * np.pi * 4
