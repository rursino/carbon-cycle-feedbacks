import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

def deseasonalise_old(x):

    mean_list = []
    for i in range(4):
        indices = range(i, len(x)+i, 4)
        sub = x[indices]
        mean_list.append(np.mean(sub))

    s = []
    for i in range(int(len(x)/4)):
        for j in mean_list:
            s.append(j)
    s = np.array(s)

    return x - (s-np.mean(s))

def deseasonalise_new(x):
    SI = []
    for index in range(0, len(x), 12):
        mean_year = np.mean(x[index : index + 12])
        for sub_index in range(index, index + 12):
            SI.append(x[sub_index] / mean_year)
    SI = np.array(SI)

    mean_SI = []
    for i in range(12):
        SI_indices = range(i, len(x) + i, 12)
        mean_SI.append(np.mean(SI[SI_indices]))
    mean_SI = np.tile(mean_SI , int(len(x) / 12))

    return x / mean_SI

def seasonal_dummy(x, s=12):



month = xr.open_dataset('./../../../output/inversions/spatial/output_all/JENA_s76/month.nc').Earth_Land.values
year = xr.open_dataset('./../../../output/inversions/spatial/output_all/JENA_s76/year.nc').Earth_Land.values

ss = deseasonalise_new(month)

plt.plot(month)
plt.plot(range(len(ss)), ss*12)
plt.plot(range(0, len(ss), 12), year)
