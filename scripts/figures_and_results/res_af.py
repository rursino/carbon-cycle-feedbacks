""" IMPORTS """
from core import AirborneFraction

from importlib import reload
reload(AirborneFraction);

import pandas as pd
import xarray as xr
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import stats

import os


""" INPUTS """
DIR = './../../'
OUTPUT_DIR = DIR + 'output/'
INV_DIRECTORY = "./../../output/inversions/spatial/output_all/"
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"

GCP = pd.read_csv(DIR + './data/GCP/budget.csv', index_col='Year')

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

temp_zero = {}
for timeres in temp:
    temp_zero[timeres] = xr.Dataset(
        {key: (('time'), np.zeros(len(temp[timeres][key]))) for key in ['Earth', 'South', 'Tropical', 'North']},
        coords={'time': (('time'), temp[timeres].time)}
    )


invf_models = os.listdir(INV_DIRECTORY)
invf_uptake = {'year': {}, 'month': {}}
for timeres in invf_uptake:
    for model in invf_models:
        model_dir = INV_DIRECTORY + model + '/'
        invf_uptake[timeres][model] = xr.open_dataset(model_dir + f'{timeres}.nc')

trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in trendy_models}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in trendy_models}
    }
}


def original_af():
    c_atm = GCP['atmospheric growth']
    fossil = GCP['fossil fuel and industry']
    luc = GCP['land-use change emissions']

    return c_atm / (fossil + luc)

def gcp_original_af(Ul, Uo):

    fossil = GCP['fossil fuel and industry']
    luc = GCP['land-use change emissions']

    af = 1 + (Ul + Uo) / (fossil + luc)

    return af.mean()

def inv_original_af(Udf):

    index = pd.to_datetime(Udf.time.values).year
    start, end = index[0], index[-1]

    fossil = GCP['fossil fuel and industry'].loc[start:end]
    luc = GCP['land-use change emissions'].loc[start:end]

    Ul = Udf.Earth_Land.values
    Uo = Udf.Earth_Ocean.values

    af = 1 + (Ul + Uo) / (fossil + luc)

    return af.mean()

def trendy_original_af(Ul, Uo):

    fossil = GCP['fossil fuel and industry'].loc[1960:2017]
    luc = GCP['land-use change emissions'].loc[1960:2017]

    af = 1 + (Ul + Uo) / (fossil + luc)

    return af.mean()

""" RESULTS """
def af_dataframe(data):
    af_df = pd.Series(data.airborne_fraction())
    af_df['alpha_5'] = af_df['alpha_mean'] - af_df['alpha_std'] * 1.645
    af_df['alpha_95'] = af_df['alpha_mean'] + af_df['alpha_std'] * 1.645

    return af_df


""" EXECUTIONS """
GCPdf = AirborneFraction.GCP(co2['year'], temp['year'])

GCPdf.airborne_fraction()

pd.Series(GCPdf.airborne_fraction())
GCP_window = pd.DataFrame(GCPdf.window_af())
GCP_window
stats.linregress(range(len(GCP_window.index)), GCP_window.af)


INVdf = AirborneFraction.INVF(co2['year'], temp['year'], invf_uptake['year'])
af_dataframe(INVdf)
inv_window = pd.DataFrame(INVdf.window_af())
inv_window['alpha_cf'] = inv_window['alpha_std'] * 1.645
inv_window['alpha_5'] = inv_window['alpha_mean'] - inv_window['alpha_cf']
inv_window['alpha_95'] = inv_window['alpha_mean'] + inv_window['alpha_cf']
inv_window
stats.linregress(range(len(inv_window.index)), inv_window.af)


TRENDYdf = AirborneFraction.TRENDY(co2['year'], temp['year'], trendy_uptake)
TRENDYdf.airborne_fraction()
af_dataframe(TRENDYdf)
TRENDY_window = pd.DataFrame(TRENDYdf.window_af())
TRENDY_window['alpha_cf'] = TRENDY_window['alpha_std'] * 1.645
TRENDY_window['alpha_5'] = TRENDY_window['alpha_mean'] - TRENDY_window['alpha_cf']
TRENDY_window['alpha_95'] = TRENDY_window['alpha_mean'] + TRENDY_window['alpha_cf']
TRENDY_window
stats.linregress(range(len(TRENDY_window.index)), TRENDY_window.af)

# Original AFs
oaf = original_af()

oaf.mean()
oaf.loc[1960:1969].mean()
oaf.loc[1970:1979].mean()
oaf.loc[1980:1989].mean()
oaf.loc[1990:1999].mean()
oaf.loc[2000:2009].mean()
oaf.loc[2009:2018].mean()



# GCP
original_af(GCP['land sink'], GCP['ocean sink'])

Udf.Earth_Land.sel(time=slice('1979', '2017'))


# inversions
inv_af = {}
for model in invf_uptake['year']:
    Udf = invf_uptake['year'][model]
    inv_af[model] = inv_original_af(Udf)

pd.Series(inv_af)

(Ul.loc[1960:2017] - UdfT).mean()


# TRENDY
TRENDY_af = {}
for model in trendy_uptake['S3']['year']:
    UdfT = trendy_uptake['S3']['year'][model].sel(time=slice('1960', '2017')).Earth_Land.values
    TRENDY_af[model] = trendy_original_af(UdfT, Uo.loc[1960:2017])

pd.Series(TRENDY_af)




# Inversions individual models
def inv_af_models(emission_rate=2):
    land = INVdf._feedback_parameters('Earth_Land')
    ocean = INVdf._feedback_parameters('Earth_Ocean')

    beta = (land + ocean).loc['beta'] / 2.12
    u_gamma = (land + ocean).loc['u_gamma']

    alpha = beta + u_gamma

    b = 1 / np.log(1 + emission_rate / 100)
    u = 1 - b * alpha

    time = {
        'CAMS': (1979, 2017),
        'JENA_s76': (1976, 2017),
        'JENA_s85': (1985, 2017),
        'CTRACKER': (2000, 2017),
        'JAMSTEC': (1996, 2017),
        'Rayner': (1992, 2012)
    }

    af = pd.DataFrame(time).T.sort_index()
    af['AF'] = (1 / u).values
    af['alpha'] = alpha
    af.columns = ['start', 'end', 'AF', 'alpha']

    return af

    return pd.DataFrame({'AF': af, 'alpha': alpha})

inv_af_models()

inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']]['AF']
inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']].mean()['AF']
inv_af_models().loc[['CTRACKER', 'JAMSTEC', 'Rayner']].std()['AF']

inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']]['AF']
inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']].mean()['AF']
inv_af_models().loc[['CAMS', 'JENA_s76', 'JENA_s85']].std()['AF']


# TRENDY individual models
def trendy_af_models(emission_rate=2):
    land = TRENDYdf._feedback_parameters('Earth_Land')

    ocean = GCPdf._feedback_parameters()['ocean']
    ocean.index = ['beta', 'gamma']
    ocean['beta'] /= 2.12
    ocean['u_gamma'] = ocean['gamma'] * 0.015 / 2.12 / 1.93

    params = land.add(ocean, axis=0)
    beta = params.loc['beta']
    u_gamma = params.loc['u_gamma']

    alpha = beta + u_gamma

    b = 1 / np.log(1 + emission_rate / 100)
    u = 1 - b * alpha

    return pd.DataFrame({'AF': 1 / u, 'alpha': alpha})

trendy_af_models()['alpha'].mean()







""" FIGURES """
# def GCP_window(save=False):
#     af_results, alpha = GCPdf.window_af()
#     emission_rate = 0.02
#
#     fig = plt.figure(figsize=(14,9))
#
#     ax1 = fig.add_subplot(212)
#     index = np.array([int(i) for i in alpha.index])
#     ax1.bar(index - 3, alpha.values, width=3)
#     ax1.axhline(linestyle='--', alpha=0.5, color='k')
#     ax1.axhline(emission_rate, linestyle='--', alpha=0.5, color='r')
#     ax1.set_ylabel(r'$\alpha$ (yr$^{-1}$)', fontsize=20, labelpad=15)
#
#     ax2 = fig.add_subplot(211)
#     x = np.array([int(i) for i in af_results.index])
#     y = af_results.values
#     ax2.bar(x, y,
#             # yerr=1.645*af_results['std'].values,
#             width=3
#            )
#
#     for i, j in enumerate(y):
#         ax2.text(x[i]+2, 0.1, f'{j:.2f}', color='blue', fontweight='bold')
#     ax2.axhline(linestyle='--', alpha=0.5, color='k')
#
#     no_outliers = [np.median(y) - 1.5 * stats.iqr(y), np.median(y) + 1.5 * stats.iqr(y)]
#     y_max = y[(y > no_outliers[0]) & (y < no_outliers[1])]
#
#     delta = 0.5
#     ax2.set_xlim([min(x) - 5 , max(x) + 5])
#     ax2.set_ylim([y.min() - delta , y_max.max() + delta])
#
#     ax2.set_ylabel(r'$\Lambda$', fontsize=20, labelpad=15)
#
# def INVF_window(save=False):
#     af_results, alpha = INVdf.window_af()
#     emission_rate = 0.02
#
#     fig = plt.figure(figsize=(14,9))
#
#     ax1 = fig.add_subplot(212)
#     ax1.bar(alpha.mean(axis=1).index - 3, alpha.mean(axis=1).values, width=3,
#             yerr=alpha.std(axis=1).values)
#     ax1.axhline(linestyle='--', alpha=0.5, color='k')
#     ax1.axhline(emission_rate, linestyle='--', alpha=0.5, color='r')
#     ax1.set_ylabel(r'$\alpha$ (yr$^{-1}$)', fontsize=20, labelpad=15)
#
#     ax2 = fig.add_subplot(211)
#     x = af_results['mean'].index
#     y = af_results['mean'].values
#     ax2.bar(x, y,
#             yerr=1.645*af_results['std'].values,
#             width=3
#            )
#     for i, j in enumerate(y):
#         ax2.text(x[i]+2, 0.1, f'{j:.2f}', color='blue', fontweight='bold')
#     ax2.axhline(linestyle='--', alpha=0.5, color='k')
#
#     no_outliers = [np.median(y) - 1.5 * stats.iqr(y), np.median(y) + 1.5 * stats.iqr(y)]
#     y_max = y[(y > no_outliers[0]) & (y < no_outliers[1])]
#
#     delta = 0.5
#     ax2.set_xlim([min(x) - 5 , max(x) + 5])
#     # ax2.set_ylim([y.min() - delta , y_max.max() + delta])
#
#     ax2.set_ylabel(r'$\Lambda$', fontsize=20, labelpad=15)
#
# def TRENDY_window(save=False):
#     af_results, alpha = TRENDYdf.window_af()
#     emission_rate = 0.02
#
#     fig = plt.figure(figsize=(14,9))
#
#     ax1 = fig.add_subplot(212)
#     ax1.bar(alpha.mean(axis=1).index - 3, alpha.mean(axis=1).values, width=3,
#             yerr=alpha.std(axis=1).values)
#     ax1.axhline(linestyle='--', alpha=0.5, color='k')
#     ax1.axhline(emission_rate, linestyle='--', alpha=0.5, color='r')
#     ax1.set_ylabel(r'$\alpha$ (yr$^{-1}$)', fontsize=20, labelpad=15)
#
#     ax2 = fig.add_subplot(211)
#     x = af_results['mean'].index
#     y = af_results['mean'].values
#     ax2.bar(x, y,
#             yerr=1.645*af_results['std'].values,
#             width=3
#            )
#     for i, j in enumerate(y):
#         ax2.text(x[i]+2, 0.1, f'{j:.2f}', color='blue', fontweight='bold')
#     ax2.axhline(linestyle='--', alpha=0.5, color='k')
#
#     no_outliers = [np.median(y) - 1.5 * stats.iqr(y), np.median(y) + 1.5 * stats.iqr(y)]
#     y_max = y[(y > no_outliers[0]) & (y < no_outliers[1])]
#
#     delta = 0.5
#     ax2.set_xlim([min(x) - 5 , max(x) + 5])
#     ax2.set_ylim([y.min() - delta , y_max.max() + delta])
#
#     ax2.set_ylabel(r'$\Lambda$', fontsize=20, labelpad=15)
