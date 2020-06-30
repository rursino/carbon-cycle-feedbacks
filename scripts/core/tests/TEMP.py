""" Test the accuracy of the output from the execution of the
TEMP.spatial_averaging function.
"""

""" IMPORTS """
import sys
sys.path.append("./../core/")
import TEMP
import numpy as np

from importlib import reload
reload(TEMP)


""" INPUTS """
dir = "./../../data/temp/crudata/"
fname = dir + "CRUTEM.4.6.0.0.anomalies.nc"

df = TEMP.SpatialAve(fname)

test = df.data.temperature_anomaly.sel(time = "2010-12")
lat = test.latitude.values
vals = test.values.squeeze()

np.nanmean(vals[lat > 30])

res23 = df.spatial_averaging(
                start_time = "2010-12",
                end_time = "2011-01",
                lat_split = 23
                )
res30 = df.spatial_averaging(
                start_time = "2010-12",
                end_time = "2011-01",
                lat_split = 30
                )

res23
res30
t = [res23.South.values.squeeze(), res23.North.values.squeeze(), res23.Tropical.values.squeeze()]
np.average(t, weights=[0.25,0.5,0.25])


len(lat[lat<-30])
