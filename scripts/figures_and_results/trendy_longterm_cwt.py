""" Produce a long-term TRENDY Cascading Window Trend and figures.
"""

""" IMPORTS """
import pandas as pd
import xarray as xr
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"
SPATIAL_DIRECTORY = "./../../output/TRENDY/spatial/mean_all/"

uptake = {
    "S1": xr.open_dataset(SPATIAL_DIRECTORY + 'S1/year.nc'),
    "S3": xr.open_dataset(SPATIAL_DIRECTORY + 'S3/year.nc')
}

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

    dataframe = pd.DataFrame(alpha, index = X)

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

    ax2 = fig.add_axes((0.1,0.1,1.6,0))
    ax2.yaxis.set_visible(False) # hide the yaxis

    new_tick_locations = np.array(np.linspace(0.01, 0.99, 18))

    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(x_co2.index[:-window_size:15])
    ax2.set_xlabel(r"Modified x-axis: $1/(1+X)$", fontsize=14)

    return dataframe

df = cascading_window_trend(time=('1820', '2016'), window_size=50)


plt.plot(df['alpha_S1_uptake'].values)
plt.plot((df['alpha_S3_uptake'].values - df['alpha_S1_uptake'].values))
plt.legend([r'$\beta$', r'$u_{\gamma}$'])
