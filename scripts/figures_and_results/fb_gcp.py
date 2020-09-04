""" IMPORTS """
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

from statsmodels import api as sm


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
        'ocean': feedback_window_regression(Ul, C, T)
    }

    fig = plt.figure(figsize=(14,9))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    bar_width = 4

    for subplot, sink in zip(['211', '212'], ['land', 'ocean']):
        ax = fig.add_subplot(subplot)

        x_beta = [int(year) - bar_width / 2 for year in df[sink].index]
        x_ugamma = [int(year) + bar_width / 2 for year in df[sink].index]
        y = df[sink]

        ax.bar(x_beta, y['beta'] / 2.12, width=bar_width, color='green')
        ax.bar(x_ugamma, y['u_gamma'], width=bar_width, color='red')
        ax.legend([r'$\beta$', r'$u_{\gamma}$'], fontsize=12)

    axl.set_xlabel('First year of 10-year window', fontsize=18, labelpad=15)
    axl.set_ylabel('Feedback parameter ($yr^{-1}$)', fontsize=18, labelpad=15)


""" EXECUTION """
latex_fb_gcp()[['beta', 'u_gamma']].sum(axis=1)
print(latex_fb_gcp())

fb_gcp_decade()



# ylim_min = (np
#     .array((df['land'][['beta', 'u_gamma']].min().values,
#             df['ocean'][['beta', 'u_gamma']].min().values))
#     .min()
# )
# ylim_max = (np
#     .array((df['land'][['beta', 'u_gamma']].max().values,
#             df['ocean'][['beta', 'u_gamma']].max().values))
#     .max()
# )
# ylim = [ylim_min, ylim_max]
