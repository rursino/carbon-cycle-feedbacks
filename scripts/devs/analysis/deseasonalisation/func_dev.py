import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from statsmodels import api as sm
from scipy import signal

def deseasonalise_old(x):

    mean_list = []
    for i in range(12):
        indices = range(i, len(x)+i, 12)
        sub = x[indices]
        mean_list.append(np.mean(sub))

    s = []
    for i in range(int(len(x)/12)):
        for j in mean_list:
            s.append(j)
    s = np.array(s)

    return x - (s-np.mean(s))

def deseasonalise_new(x):

    fs = 12
    fc = 365/667

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'low')

    return signal.filtfilt(b, a, x)


month = xr.open_dataset('./../../../../output/inversions/spatial/output_all/JENA_s76/month.nc').Earth_Land
year = xr.open_dataset('./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc').Earth_Land

x = month.values
xd = deseasonalise_new(x)

plt.plot(x)
plt.plot(deseasonalise_old(x))
plt.plot(deseasonalise_new(x))
