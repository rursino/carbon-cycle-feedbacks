""" Devs on the Analysis section for TRENDY mean_all datasets.
"""


""" IMPORTS """
import xarray as xr
import matplotlib.pyplot as plt


""" INPUTS """
DIR = './../../../output/TRENDY/spatial/mean_all/'


""" EXECUTION """
df1 = xr.open_dataset(DIR + 'S1/decade.nc')
df2 = xr.open_dataset(DIR + 'S3/decade.nc')=

x = df1.time.values
y1 = df1.Earth_Land.values
y2 = df2.Earth_Land.values

plt.plot(x, y1)
plt.plot(x, y2)
