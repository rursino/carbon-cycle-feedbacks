"""Development of the TEMP.py script.
"""


""" IMPORTS """
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
from core import TEMP

from importlib import reload
reload(TEMP);


""" INPUTS """
# fname = "./../../../data/temp/crudata/HadSST.3.1.1.0.median.nc"
# fname = "./../../../data/temp/crudata/HadCRUT.4.6.0.0.median.nc"
# fname = "./../../../data/temp/crudata/CRUTEM.4.6.0.0.anomalies.nc"


""" EXECUTION """
df = TEMP.SpatialAve(fname)

df.data

df.time_range("2000-01", "2000-05")

df.regional_cut((30, 90), (-180,180), "2000-01", "2000-09")

df.latitudinal_splits(23, start_time = "1994-09", end_time = "2004-05")
