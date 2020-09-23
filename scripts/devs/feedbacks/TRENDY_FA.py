""" Development of TRENDY formulation of the FeedbackAnalysis, where the
advantage of the availability of S1 and S3 datasets is taken.
"""

from core import FeedbackAnalysis

import pandas as pd
import xarray as xr
import numpy as np
from scipy import signal
from statsmodels import api as sm


def deseasonalise(x):
    """
    """

    fs = 12
    fc = 365/667

    w = fc / (fs / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'low')

    return signal.filtfilt(b, a, x)


trendy_duptake['S1']['month']['VISIT'].Earth_Land


from importlib import reload
reload(FeedbackAnalysis);

df = FeedbackAnalysis.TRENDY(co2['month'], temp['month'], trendy_duptake, 'Earth_Land')
df.regstats()['S1']['VISIT']


DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'

co2 = {
    "year": pd.read_csv(DIR + f"data/CO2/co2_year.csv", index_col=["Year"]).CO2,
    "month": pd.read_csv(DIR + f"data/CO2/co2_month.csv", index_col=["Year", "Month"]).CO2
}

temp = {
    "year": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/year.nc'),
    "month": xr.open_dataset(OUTPUT_DIR + f'TEMP/spatial/output_all/HadCRUT/month.nc')
}

trendy_models = ['VISIT', 'OCN', 'JSBACH', 'CLASS-CTEM', 'CABLE-POP']
trendy_uptake = {
    "S1": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S1_nbp/month.nc')
        for model_name in trendy_models if model_name != "LPJ-GUESS"}
    },
    "S3": {
        "year": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/year.nc')
        for model_name in trendy_models},
        "month": {model_name : xr.open_dataset(OUTPUT_DIR + f'TRENDY/spatial/output_all/{model_name}_S3_nbp/month.nc')
        for model_name in trendy_models if model_name != "LPJ-GUESS"}
    }
}

trendy_duptake = {'S1': {'month': {}}, 'S3': {'month': {}}}
for sim in trendy_uptake:
    df_models = trendy_uptake[sim]['month']
    for model in trendy_uptake[sim]['month']:
        trendy_duptake[sim]['month'][model] = xr.Dataset(
            {key: (('time'), deseasonalise(df_models[model][key].values)) for
            key in ['Earth_Land', 'South_Land', 'North_Land', 'Tropical_Land']},
            coords={'time': (('time'), df_models[model].time.values)}
        )

temp_zero = {}
for timeres in temp:
    temp_zero[timeres] = xr.Dataset(
        {key: (('time'), np.zeros(len(temp[timeres][key]))) for key in ['Earth', 'South', 'Tropical', 'North']},
        coords={'time': (('time'), temp[timeres].time)}
    )

phi, rho = 0.015 / 2.12, 1.93


def feedback_regression(timeres, variable):
    uptake = trendy_duptake if timeres == "month" else trendy_uptake
    start, end = 1960, 2017
    reg_models = {}
    model_stats = {}
    for simulation in ['S1', 'S3']:
        model_names = uptake[simulation][timeres].keys()

        sim_reg_models = {}
        sim_model_stats = {
                'r_squared': [],
                't_values_beta': [],
                't_values_gamma': [],
                'p_values_beta': [],
                'p_values_gamma': [],
                'mse_total': [],
                'nobs': []
            }

        for model_name in model_names:
            C = co2[timeres].loc[start:end]
            T = temp[timeres].sel(time=slice(str(start), str(end)))
            if simulation == 'S1':
                U = uptake['S1'][timeres][model_name]
            elif simulation == 'S3':
                U = uptake['S3'][timeres][model_name] - uptake['S1'][timeres][model_name]
            U = U.sel(time=slice(str(start), str(end)))

            df = pd.DataFrame(data = {
                                    "C": C,
                                    "U": U[variable],
                                    "T": T['Earth']
                                    }
                                   )

            if simulation == 'S1':
                x_var = 'C'
                other_var = 'T'
            elif simulation == 'S3':
                x_var = 'T'
                other_var = 'C'
            X = sm.add_constant(df[x_var])
            Y = df["U"]
            reg_model = sm.OLS(Y, X).fit()
            reg_model_params = reg_model.params
            reg_model_params[other_var] = 0
            sim_reg_models[model_name] = (reg_model_params
                                            .loc[[x_var, other_var]]
                                            .sort_index()
                                         )
            sim_model_stats['r_squared'].append(reg_model.rsquared)
            if simulation == 'S1':
                sim_model_stats['t_values_beta'].append(reg_model.tvalues.loc['C'])
                sim_model_stats['p_values_beta'].append(reg_model.pvalues.loc['C'])
                sim_model_stats['t_values_gamma'].append(np.nan)
                sim_model_stats['p_values_gamma'].append(np.nan)
            elif simulation == 'S3':
                sim_model_stats['t_values_beta'].append(np.nan)
                sim_model_stats['p_values_beta'].append(np.nan)
                sim_model_stats['t_values_gamma'].append(reg_model.tvalues.loc['T'])
                sim_model_stats['p_values_gamma'].append(reg_model.pvalues.loc['T'])
            sim_model_stats['mse_total'].append(reg_model.mse_total)
            sim_model_stats['nobs'].append(reg_model.nobs)

        df = pd.DataFrame(sim_reg_models)
        df.index = ['beta', 'gamma']
        reg_models[simulation] = df
        reg_models[simulation].loc['beta'] /= 2.12
        reg_models[simulation].loc['u_gamma'] = reg_models[simulation].loc['gamma'] * phi / rho

        model_stats[simulation] = pd.DataFrame(sim_model_stats, index=model_names)

    return reg_models, model_stats


feedback_regression('month', 'Earth_Land')[0]
