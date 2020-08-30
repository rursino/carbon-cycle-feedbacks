import pickle
import matplotlib.pyplot as plt
import pandas as pd

import sys
from core import inv_flux as invf
from core import GCP_flux as GCPf

from importlib import reload
reload(invf)
reload(GCPf)

# Inversion
fname = "./../../../output/inversions/raw/output_all/JENA_s76_all/year.pik"

df = invf.Analysis(pickle.load(open(fname, 'rb')))

variable = "Earth_Land"
df.rolling_trend(variable, plot=True);
plt.title(f"25-Year Rolling Trend: {variable}", fontsize=28)

#GCP
land = GCPf.Analysis("land sink")
ocean = GCPf.Analysis("ocean sink")

land.rolling_trend(plot=True);
plt.title(f"25-Year Rolling Trend: GCP Land", fontsize=28)

ocean.rolling_trend(plot=True);
plt.title(f"25-Year Rolling Trend: GCP Ocean", fontsize=28)

#model evaluation
me = invf.ModelEvaluation(pickle.load(open(fname, 'rb')))

me.regress_rolling_trend_to_GCP("land", 25, True)
me.regress_rolling_trend_to_GCP('ocean', 25, True)
