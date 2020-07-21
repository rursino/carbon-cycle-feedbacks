""" Test to see if deseasonalisation is working correctly by determining if
the frequency 12 months is removed from the deseasonalised timeseries.
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

import sys
sys.path.append("./../../core")
import TRENDY_flux as TRENDYf

from importlib import reload
reload(TRENDYf)

fname = f"./../../../output/TRENDY/spatial/output_all/CABLE-POP_S1_nbp/month.nc"

ds = xr.open_dataset(fname)
df = TRENDYf.Analysis(ds)

freqs, spec = signal.welch(ds.Earth_Land.values, fs=1/2)
plt.semilogy(1/freqs, spec)
freqs, spec = signal.welch(df.deseasonalise("Earth_Land"), fs=1/2)
plt.semilogy(1/freqs, spec)
plt.gca().invert_xaxis()
plt.xlim([50,0])

""" Blue is original, orange is deseasonalised.
Can see that the deseasonalisation works since it removes power at 12 months
and at other frequencies in sync with 12 months.
"""
