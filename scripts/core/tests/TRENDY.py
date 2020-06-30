""" Test the accuracy of the output from the execution of the
TRENDY.spatial_integration function.
"""

""" IMPORTS """
import sys
sys.path.append("./../core/")
import TRENDY_flux as Tflux
import inv_flux as invf
import numpy as np

from importlib import reload
reload(Tflux)


""" INPUTS """
fname = "./../../data/TRENDY/models/LPJ-GUESS/S0/LPJ-GUESS_S0_cVeg.nc"

df = Tflux.SpatialAgg(fname)
res = df.spatial_integration()

res


def scalar_earth_area(minlat, maxlat, minlon, maxlon):
    """Returns the area of earth in the defined grid box."""

    r_earth = 6.371e6 # Radius of Earth
    dtor = np.pi / 180. # conversion from degrees to radians.

    diff_lon = np.unwrap((minlon, maxlon), discont=360.+1e-6)

    return 2. * np.pi * r_earth**2 * (np.sin(dtor * maxlat) -
    np.sin(dtor * minlat)) * (diff_lon[1] - diff_lon[0]) / 360.


def earth_area(minlat, maxlat, minlon, maxlon):
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


""" TEST """
flux = df.data.cVeg.sel(time="1996").values.squeeze()
eag = earth_area_grid(df.data.latitude, df.data.longitude)

lat = df.data.latitude
sink = (flux * eag)[(lat > -30) & (lat < 30)]
(np.nansum(sink)*1e-15)

res
res.isel(time=-23)

fname1 = "./../../data/inversions/fco2_JAMSTEC-V1-2-2018_June2018-ext3_1996-2017_monthlymean_XYT.nc"

df1 = invf.SpatialAgg(fname1)
res1 = df1.spatial_integration()

res1.resample(time="Y").sum().isel(time=-1)
