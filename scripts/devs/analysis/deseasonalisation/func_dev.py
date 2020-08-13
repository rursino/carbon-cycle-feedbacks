import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from statsmodels import api as sm

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


month = xr.open_dataset('./../../../../output/inversions/spatial/output_all/JENA_s76/month.nc').Earth_Land
year = xr.open_dataset('./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc').Earth_Land

df = pd.DataFrame({'Cu': month.values}, index=month.time.values)
df['month'] = df.index.month
df['t'] = range(1, len(df)+1)
df
dummy = pd.get_dummies(df, columns=['month'])

Y = dummy.Cu
X = dummy.iloc[:, 1:]

model = sm.OLS(Y, X).fit()
model.params

Yreg = np.matmul(dummy.iloc[:, 1:], model.params)

plt.plot(dummy.t * model.params['t'])

plt.plot(Y[:36])
plt.plot(Yreg[:36])


model.summary()
len(dummy)
years = int(len(dummy) / 12)
df['seasonal_params'] = np.tile(model.params, years).__len__()
df
