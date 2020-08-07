""" Convert original output from FeedBackAnalysis (txt files) for application
to the FeedbackOutput class.
"""

""" IMPORTS """
import os
import pandas as pd
from itertools import *
from collections import namedtuple
from tqdm import tqdm


""" INPUTS """
# Establish a list of tuples containing all combination of inputs.

DIR = './../../../output/feedbacks/'

inversions = ['CAMS', 'Rayner', 'CTRACKER', 'JAMSTEC', 'JENA_s76', 'JENA_s85']
TRENDYs_product = list(product(['CABLE-POP', 'JSBACH',
                                'CLASS-CTEM', 'OCN', 'LPJ-GUESS'],
                               ['S1', 'S3'],
                               ['nbp']
                              )
                      )
TRENDYs = [f'{i[0]}_{i[1]}_{i[2]}' for i in TRENDYs_product]

regions = ["Earth", "South", "North", "Tropical"]

tempsink_invs = list((input[0][0], input[1]+input[0][1]) for
                input in product(zip(['CRUTEM', 'HadSST'], ['_Land', '_Ocean']),
                                 regions))
tempsink_TRENDYs = list(product(['CRUTEM'],
                                (region + '_Land' for region in regions)
                               )
                       )

inputs = (list(product(inversions, tempsink_invs)) +
         list(product(TRENDYs, tempsink_TRENDYs))
         )


""" FUNCTIONS """
def extract_param_vals(readLines, index):
    """ Extract information about the paramater values including the upper and
    lower limits of confidence intervals.
    """

    param = readLines[index].split()
    param_name = param[0]
    median = float(param[1])
    left = float(param[2])
    right = float(param[3])

    Param = namedtuple(param_name, ['median',
                                    'confint_left',
                                    'confint_right'
                                   ])

    return Param(median, left, right)

def extract_model_vals(readLines):
    """ Extract other statistical information in the readLines list.
    """

    # Extract r-value.
    r = float(readLines[22].split()[1])

    # Extract t-values for each parameter.
    t_vals = readLines[26].split()

    T_tuple = namedtuple('T', ['const', 'beta', 'gamma'])
    t = T_tuple(float(t_vals[1]), float(t_vals[2]), float(t_vals[3]))

    # Extract p-values for each parameter.
    p_vals = readLines[27].split()

    P_tuple = namedtuple('P', ['const', 'beta', 'gamma'])
    p = P_tuple(float(p_vals[1]), float(p_vals[2]), float(p_vals[3]))

    # Extract MSE and nobs.
    MSE = float(readLines[29].split()[2])
    nobs = int(readLines[31].split()[-1])

    return r, t, p, MSE, nobs

def params_dataframe(fnames, time_resolution):
    """ Insert information about the parameters and confidence intervals, and
    with all time ranges into one dataframe.
    """

    params_dict = {
        "index": [],
        "const_left": [],
        "const_median": [],
        "const_right": [],
        "beta_left": [],
        "beta_median": [],
        "beta_right": [],
        "gamma_left": [],
        "gamma_median": [],
        "gamma_right": [],
    }

    for fname in fnames:
        # Read text file and extract lines.
        with open(destination + fname, 'r') as f:
            readLines = f.readlines()

        timeres = readLines[8].split()[-1]
        if time_resolution != timeres:
            continue

        index = readLines[9].split()[2]
        heading, confint_left, confint_right = readLines[16].split()
        const = extract_param_vals(readLines, 18)
        beta = extract_param_vals(readLines, 19)
        gamma = extract_param_vals(readLines, 20)
        r, T, P, MSE, nobs = extract_model_vals(readLines)

        params_dict['index'].append(index)
        params_dict['const_median'].append(const.median)
        params_dict['const_left'].append(const.confint_left)
        params_dict['const_right'].append(const.confint_right)
        params_dict['beta_median'].append(beta.median)
        params_dict['beta_left'].append(beta.confint_left)
        params_dict['beta_right'].append(beta.confint_right)
        params_dict['gamma_median'].append(gamma.median)
        params_dict['gamma_left'].append(gamma.confint_left)
        params_dict['gamma_right'].append(gamma.confint_right)

    tformat = "%Y-%m" if timeres == "month" else "%Y"
    return pd.DataFrame(params_dict).set_index('index')

def stats_dataframe(fnames, time_resolution):
    """ Insert all of the other statistical information with all time ranges
    into one dataframe.
    """

    stats_dict = {
        "index": [],
        "r": [],
        "T_const": [],
        "T_beta": [],
        "T_gamma": [],
        "P_const": [],
        "P_beta": [],
        "P_gamma": [],
        "MSE": [],
        "nobs": [],
    }

    for fname in fnames:
        # Read text file and extract lines.
        with open(destination + fname, 'r') as f:
            readLines = f.readlines()

        timeres = readLines[8].split()[-1]
        if time_resolution != timeres:
            continue

        index = readLines[9].split()[2]
        r, T, P, MSE, nobs = extract_model_vals(readLines)

        stats_dict['index'].append(index)
        stats_dict['r'].append(r)
        stats_dict['T_const'].append(T.const)
        stats_dict['T_beta'].append(T.beta)
        stats_dict['T_gamma'].append(T.gamma)
        stats_dict['P_const'].append(P.const)
        stats_dict['P_beta'].append(P.beta)
        stats_dict['P_gamma'].append(P.gamma)
        stats_dict['MSE'].append(MSE)
        stats_dict['nobs'].append(nobs)

    tformat = "%Y-%m" if timeres == "month" else "%Y"
    return pd.DataFrame(stats_dict).set_index('index')


""" EXECUTION """
for input in tqdm(inputs):
    destination = DIR + f'{input[0]}/{input[1][0]}/{input[1][1]}/'

    # Extract all file paths and remove directories from list if included.
    fnames = os.listdir(destination)
    if 'csv_output' in fnames:
        fnames.remove('csv_output')

    # Create dataframes to be saved.
    dataframes = (
        ('params_year', params_dataframe(fnames, 'year')),
        ('params_month', params_dataframe(fnames, 'month')),
        ('stats_year', stats_dataframe(fnames, 'year')),
        ('stats_month', stats_dataframe(fnames, 'month'))
    )

    # Save dataframes as csvs.
    try:
        for df_name, df in dataframes:
            df.to_csv(destination + 'csv_output/' + df_name + '.csv')
    except OSError:
        os.mkdir(destination + 'csv_output/')
        for df_name, df in dataframes:
            df.to_csv(destination + 'csv_output/' + df_name + '.csv')
