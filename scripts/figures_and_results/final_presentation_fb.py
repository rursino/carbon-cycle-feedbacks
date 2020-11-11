""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import numpy as np
import xarray as xr
from statsmodels import api as sm

import seaborn as sns
sns.set_style('darkgrid')

import sys
from copy import deepcopy
sys.path.append('../core')
import GCP_flux as GCPf
import inv_flux as invf
import trendy_flux as TRENDYf
import FeedbackAnalysis

import os

from importlib import reload
reload(GCPf);
reload(invf);
reload(TRENDYf);


""" INPUTS """
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
TRENDY_DIRECTORY = "./../../output/TRENDY/spatial/output_all/"
INPUT_DIRECTORY = './../../data/'
DIR = './../../'
OUTPUT_DIR = DIR + 'output/'

co2 = pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2
temp = xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc')

temp_zero = xr.Dataset(
    {key: (('time'), np.zeros(len(temp[key]))) for key in ['Earth', 'South', 'Tropical', 'North']},
    coords={'time': (('time'), temp.time)}
)

invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}, 'winter': {}, 'summer': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')


trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
        "S1": {'year':
                {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
                for model_name in trendy_models}
                },
        "S3": {'year':
                {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
                for model_name in trendy_models}
            }
}

gcp_uptake = pd.read_csv(INPUT_DIRECTORY + 'GCP/budget.csv',
                        index_col='Year',
                        usecols=[0,4,5]
                       )
Ul = gcp_uptake['land sink']
Uo = gcp_uptake['ocean sink']

C = pd.read_csv(INPUT_DIRECTORY + 'CO2/co2_year.csv',index_col='Year').CO2[2:]
T = (xr
        .open_dataset('./../../output/TEMP/spatial/output_all/HadCRUT/year.nc')
        .sel(time=slice("1959", "2018"))
    )
T= T.Earth

phi, rho = 0.015 / 2.12, 1.93


""" FUNCTIONS """
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
        index = co2.index
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


""" PARAMS """

df_invf_land = FeedbackAnalysis.INVF(
                            co2,
                            temp,
                            invf_uptake['year'],
                            "Earth_Land"
                            ).params()

df_invf_ocean = FeedbackAnalysis.INVF(
                            co2,
                            temp,
                            invf_uptake['year'],
                            "Earth_Ocean"
                            ).params()

invf_params = {
    'land': {
        'beta': df_invf_land['beta'],
        'u_gamma': df_invf_land['u_gamma']
    },
    'ocean': {
        'beta': df_invf_ocean['beta'],
        'u_gamma': df_invf_ocean['u_gamma']
    }
}

df_trendy = FeedbackAnalysis.TRENDY(
                            'year',
                            co2,
                            temp,
                            trendy_uptake,
                            "Earth_Land"
                            )

trendy_params = {
    'beta': df_trendy.params()['S1']['beta'],
    'u_gamma': df_trendy.params()['S3']['u_gamma']
}


gcp_fb = {
    'land': feedback_window_regression(Ul, C, T),
    'ocean': feedback_window_regression(Uo, C, T)
}

gcp_params = {var: {param: gcp_fb[var][param] for param in ['beta', 'u_gamma']} for var in ['land', 'ocean']}


""" FIGURE """
def fb_plot():
    fig = plt.figure(figsize=(12,14))
    ax = {}
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    bar_width = 2

    fb_gcp_vals = {}

    zip_list = zip(
        ['411', '412', '413', '414'],
        [
            ('land', 'beta', r'$\beta_{L}$'),
            ('land', 'u_gamma', r'$u_{\gamma,L}$'),
            ('ocean', 'beta', r'$\beta_{O}$'),
            ('ocean', 'u_gamma', r'$u_{\gamma,O}$')
        ]
    )

    trendy_list = zip(
        ['411', '412'],
        ['beta', 'u_gamma']
    )

    for subplot, parameter in zip_list:
        var, param, label = parameter
        ax[subplot] = fig.add_subplot(subplot)

        gcp_x = np.arange(1960, 2020, 10)
        gcp = gcp_params[var][param]

        inv_x = np.arange(1980, 2020, 10)
        inv = invf_params[var][param].mean(axis=1).values
        inv_std = invf_params[var][param].std(axis=1).values

        ax[subplot].bar(gcp_x - bar_width, gcp, width=bar_width, color='black')
        ax[subplot].bar(inv_x, inv, width=bar_width, yerr=inv_std, color='cyan')

        ax[subplot].set_ylim([-0.16, 0.11]) if var == 'land' else ax[subplot].set_ylim([-0.04, 0.02])

        ax[subplot].set_xlabel(label, fontsize=24, labelpad=10)
        ax[subplot].xaxis.set_label_position('top')

    for subplot, param in trendy_list:

        trendy_x = np.arange(1960, 2020, 10)
        trendy = trendy_params[param].mean(axis=1).values
        trendy_std = trendy_params[param].std(axis=1).values

        ax[subplot].bar(trendy_x + bar_width, trendy, width=bar_width, yerr=trendy_std, color='red')

    axl.set_ylabel('Feedback parameter ($yr^{-1}$)', fontsize=28, labelpad=40)
    ax['411'].legend(['GCP', 'Inversions', 'TRENDY'])

    for subplot in ax:
        ax[subplot].tick_params(axis="x", labelsize=14)
        ax[subplot].tick_params(axis="y", labelsize=14)

    fig.tight_layout(pad=3.0)


fb_plot()
