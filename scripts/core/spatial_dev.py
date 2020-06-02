import numpy as np
from itertools import *

lat = np.arange(-90,90,0.5)
lat_split = 30

lat_conditions = (
True, lat < -lat_split,
(lat>-lat_split) & (lat<lat_split), lat>lat_split
)

earth_land_sink = lat*7

variables = ["el", "sl", "tl", "nl"]

values = []

conditions = iter(lat_conditions)
for var in variables:
    sum = np.sum(1e-15*earth_land_sink)[next(conditions)]
    values.append(sum)
