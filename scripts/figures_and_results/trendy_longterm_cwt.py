""" Produce a long-term TRENDY Cascading Window Trend and figures.
"""

""" IMPORTS """
import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import signal


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/TRENDY/spatial/mean_all/"

uptake = {
    "S1": xr.open_dataset(SPATIAL_DIRECTORY + 'S1/year.nc'),
    "S3": xr.open_dataset(SPATIAL_DIRECTORY + 'S3/year.nc')
}

# Bandpass
w = (1 / 1.827) / (12 / 2)
b, a = signal.butter(5, w, 'low')

vars = ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']
Ubp = {}
for sim in ["S1", "S3"]:
    Ubp[sim] = xr.Dataset(
        {var: (('time'), signal.filtfilt(b, a, uptake[sim][var])) for var in vars},
        coords={'time': (('time'), uptake[sim].time.values)}
        )

co2 = pd.read_csv("./../../data/co2/co2_longterm.csv",
                  usecols=[2,3],
                  index_col="Year")
co2 = co2.loc[1701:2016]


""" FUNCTIONS """
def cascading_window_trend(variable="Earth_Land", time=('1750','2016'), window_size=25):
    x_co2 = co2.loc[time[0]:time[1]]
    df = pd.DataFrame(
        {
        "co2": x_co2.values.squeeze(),
        "S1_uptake": uptake["S1"][variable].sel(time=slice(*time)),
        "S3_uptake": uptake["S3"][variable].sel(time=slice(*time))
        },
        index = x_co2.index
    )

    X = x_co2['CO2'].values[:-window_size]
    alpha = {}
    for sim in ['S1_uptake', 'S3_uptake']:
        Y = []
        for year in range(len(df) - window_size):
            sub_df = df[year:year + window_size+1]
            sub_X = sm.add_constant(sub_df['co2'])
            sub_Y = sub_df[sim]

            model = sm.OLS(sub_Y, sub_X).fit()
            Y.append(model.params.co2)
        alpha[f"alpha_{sim}"] = np.array(Y) * 1e3

    return pd.DataFrame(alpha, index = X).sort_index()

def trendy_longterm_cwt(dataframe, save=False):

    fig = plt.figure()
    ax1 = fig.add_axes((0.1,0.3,1.6,1))

    ax1.plot(dataframe.index, dataframe['alpha_S1_uptake'])
    ax1.plot(dataframe.index, dataframe['alpha_S3_uptake'])
    ax1.legend([r'$\alpha_{S1}$', r'$\alpha_{S3}$'], fontsize=14)

    ax1.set_title("Cascading Window 10-Year Trend: TRENDY",
                 fontsize=24, pad=20)
    ax1.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=14, labelpad=10)
    ax1.set_ylabel(r"$\alpha$   " + " (MtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

df = cascading_window_trend(time=('1900', '2016'), window_size=40)

trendy_longterm_cwt(df)

def difference(Udf):
    plt.plot(Udf['S1'].Earth_Land)
    plt.plot(Udf['S3'].Earth_Land)
    return (Udf['S3'].Earth_Land - Udf['S1'].Earth_Land).sum()

difference(uptake)
difference(Ubp)
