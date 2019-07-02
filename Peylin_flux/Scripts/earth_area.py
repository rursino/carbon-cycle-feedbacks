''' Functions to extract the area of each rectangular gridbox of the earth or a set of them. '''

import numpy
r_earth = 6.371e6 # radius of Earth

from numpy import pi
dtor = pi/180. # conversion from degrees to radians.

def scalar_earth_area( minlat, maxlat, minlon, maxlon):
    "area of earth in the defined box"
    diff_lon = numpy.unwrap((minlon,maxlon), discont=360.+1e-6)
    return 2.*pi*r_earth**2 *(numpy.sin( dtor*maxlat) -numpy.sin(dtor*minlat))*(diff_lon[1]-diff_lon[0])/360.

def earth_area( minlat, maxlat, minlon, maxlon):
    "returns grid of areas with shape=(minlon, minlat) and earth area"
    result = numpy.zeros((numpy.array(minlon).size, numpy.array(minlat).size))
    diffsinlat = 2.*pi*r_earth**2 *(numpy.sin( dtor*maxlat) -numpy.sin(dtor*minlat))
    diff_lon = numpy.unwrap((minlon,maxlon), discont=360.+1e-6)
    for i in xrange(numpy.array(minlon).size): result[ i, :] = diffsinlat *(diff_lon[1,i]-diff_lon[0,i])/360.
    return result

def earth_area_grid(lats,lons):
    """grids earth area hopefully"""
    result=numpy.zeros( (lats.size, lons.size))
    minlats = lats -0.5*(lats[1]-lats[0])
    maxlats= lats +0.5*(lats[1]-lats[0])
    minlons = lons -0.5*(lons[1]-lons[0])
    maxlons=lons +0.5*(lons[1]-lons[0])
    for i in xrange(lats.size):
        result[i,:] = scalar_earth_area(minlats[i],maxlats[i],minlons[0],maxlons[0])
    return result
