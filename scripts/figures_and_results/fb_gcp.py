""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from statsmodels import api as sm

from copy import deepcopy


""" INPUTS """
INPUT_DIRECTORY = './../../data/'
U = pd.read_csv(INPUT_DIRECTORY + 'GCP/budget.csv',
                index_col='Year',
                usecols=[0,4,5]
               )
Ul = U['land sink']
Uo = U['ocean sink']

C = pd.read_csv(INPUT_DIRECTORY + 'CO2/co2_year.csv',index_col='Year').CO2[2:]
T = (xr
        .open_dataset('./../../output/TEMP/spatial/output_all/HadCRUT/year.nc')
        .sel(time=slice("1959", "2018"))
    ).Earth

phi, rho = 0.015, 1.93

""" FUNCTIONS """
def feedback_regression(uptake, co2, temp):
    df = pd.DataFrame(
        {
            "C": co2,
            "T": temp,
            "U": uptake
        },
        index = C.index
    )

    X = sm.add_constant(df[["C", "T"]])
    Y = df["U"]

    return sm.OLS(Y, X).fit()

def latex_fb_gcp():
    fb_land = feedback_regression(Ul, C, T)
    fb_ocean = feedback_regression(Uo, C, T)

    latex_df = pd.DataFrame(
        {
            'land': fb_land.params.loc[['C', 'T']].values,
            'ocean': fb_ocean.params.loc[['C', 'T']].values
        },
        index = ['beta', 'gamma']
    ).T

    latex_df["beta"] /= 2.12
    latex_df["u_gamma"] = latex_df["gamma"] * phi / rho

    return latex_df

    return latex_df.to_latex(float_format="%.3f")

def feedback_window_regression(uptake, co2, temp):

    times = (
        ('1959', '1968'),
        ('1969', '1978'),
        ('1979', '1988'),
        ('1989', '1998'),
        ('1999', '2008'),
        ('2009', '2018'),
    )

    df = pd.DataFrame(
        {
            "C": co2,
            "T": temp,
            "U": uptake
        },
        index = C.index
    )

    params = {'beta': [], 'gamma': []}
    for time in times:
        X = sm.add_constant(df[["C", "T"]].loc[time[0]:time[1]])
        Y = df["U"].loc[time[0]:time[1]]
        model = sm.OLS(Y, X).fit()

        params['beta'].append(model.params.loc['C'])
        params['gamma'].append(model.params.loc['T'])

    dataframe = pd.DataFrame(
        params,
        index = [time[0] for time in times]
    )
    dataframe['u_gamma'] = dataframe['gamma'] * phi / rho

    return dataframe


""" FIGURES """
def fb_gcp_decade(save=False):
    df = {
        'land': feedback_window_regression(Ul, C, T),
        'ocean': feedback_window_regression(Uo, C, T)
    }

    fig = plt.figure(figsize=(12,10))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    bar_width = 4

    for subplot, sink in zip(['211', '212'], ['land', 'ocean']):
        ax[subplot] = fig.add_subplot(subplot)

        x = [int(year) for year in df[sink].index]
        x_beta = [int(year) - bar_width / 2 for year in df[sink].index]
        x_ugamma = [int(year) + bar_width / 2 for year in df[sink].index]
        y = df[sink]

        ax[subplot].bar(x_beta, y['beta'] / 2.12, width=bar_width, color='green')
        ax[subplot].bar(x_ugamma, y['u_gamma'], width=bar_width, color='red')
        ax[subplot].plot(x, y['beta'] / 2.12 + y['u_gamma'], color='black')

        ax[subplot].legend([r'$\alpha$', r'$\beta$', r'$u_{\gamma}$'],
                           fontsize=12)

    ax["211"].set_xlabel("Land", fontsize=18, labelpad=10)
    ax["211"].xaxis.set_label_position('top')
    ax["212"].set_xlabel("Ocean", fontsize=18, labelpad=10)
    ax["212"].xaxis.set_label_position('top')

    axl.set_ylabel('Feedback parameter ($yr^{-1}$)', fontsize=18, labelpad=15)


""" EXECUTION """
latex_fb_gcp()[['beta', 'u_gamma']].sum(axis=1)
print(latex_fb_gcp())

feedback_window_regression(Ul, C, T).summary()
feedback_window_regression(Uo, C, T)


fb_gcp_decade()



def resample(df):
    df.index = T.time.values
    return df.resample('10Y').mean().values

plt.bar(range(0, 19, 3),T.resample({"time": "10Y"}).mean().values)
# plt.bar(resample(C))
plt.bar(range(1, 20, 3), resample(Uo))
# plt.bar(range(2, 21, 3), resample(Ul))
plt.legend(['T', 'Uo'])
