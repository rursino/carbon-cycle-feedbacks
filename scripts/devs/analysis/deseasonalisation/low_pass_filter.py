""" Attempt to low-pass filter CO2 concentration timeseries to remove
seasonal amplitude as in Thoning et al. (1989).
"""

""" IMPORTS """
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr


""" INPUTS """
fco2m = './../../../../data/CO2/co2_month.csv'
fco2y = './../../../../data/CO2/co2_year.csv'
fuptakem = './../../../../output/inversions/spatial/output_all/CAMS/month.nc'
fuptakey = './../../../../output/inversions/spatial/output_all/CAMS/year.nc'


""" FUNCTIONS """
def cutoff(x, btype, fc, fs=12, order=5):

    if btype == "band":
        assert type(fc) == list, "fc must be a list of two values."
        fc = np.array(fc)

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(order, w, btype)

    return signal.filtfilt(b, a, x)


""" EXECUTION """
co2y = pd.read_csv(fco2y).CO2
co2m = pd.read_csv(fco2m).CO2

uptakem = xr.open_dataset(fuptakem).Earth_Land.values
uptakey = xr.open_dataset(fuptakey).Earth_Land.values

indexy = pd.read_csv(fco2y).Year
indexm = pd.read_csv(fco2m).Decimal

filt_co2m = cutoff(co2m, "low", 1/(667/365), 12)

plt.plot(indexy, co2y)
# plt.plot(indexm, co2m)
plt.plot(indexm, filt_co2m)

psd_co2m = signal.welch(co2m, fs=12)
psd_filtco2m = signal.welch(filt_co2m, fs=12)
plt.loglog(*psd_co2m)
plt.loglog(*psd_filtco2m)


filt_uptakem = cutoff(uptakem, "low", 1/(667/365), 12)

plt.plot(range(0, 468, 12), uptakey/12)
plt.plot(uptakem)

plt.plot(filt_uptakem)

psd_uptakem = signal.welch(uptakem, fs=12)
psd_filtuptakem = signal.welch(filt_uptakem, fs=12)
plt.loglog(*psd_uptakem)
plt.loglog(*psd_filtuptakem)
