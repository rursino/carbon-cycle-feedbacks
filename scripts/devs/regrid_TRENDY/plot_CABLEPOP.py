""" Plotting of TRENDY fluxes from CABLE-POP model to see if latitudes are
shifted 4 degrees north or instead need to be shifted.
"""

""" IMPORTS """
import xarray as xr
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np


""" INPUTS """
fname = './../../../data/TRENDY/models/CABLE-POP/S3/CABLE-POP_S3_nbp.nc'
earth_radius = 6.371e6

""" FUNCTIONS """
def single_time_plot(time):
    nbp = df.nbp.sel(time=time).values[0]

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.contourf(df.longitude, df.latitude, nbp)

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
df = xr.open_dataset(fname)

df.latitude
df.longitude

single_time_plot("2000-01")

vals = df.nbp.sel(time="2000-01").values[0]
vals[vals > 0] = 1
vals[np.isnan(vals)] = 0
vals
(vals * earth_area_grid(df.latitude, df.longitude)).sum() - earth_radius**2 * np.pi * 4
