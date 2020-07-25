""" Convert original output from FeedBackAnalysis (txt files) for application
to the FeedbackOutput class.
"""

""" IMPORTS """
from collections import namedtuple


""" INPUTS """
model = 'CAMS'
temp = 'CRUTEM'
time_range = '1979_1989'
region = 'Earth'
sink = 'Land' if temp == 'CRUTEM' else 'Ocean'

DIR = './../../../output/feedbacks/output_all'
destination = DIR + f'{model}/{temp}/{region}_{sink}/
fnames = os.listdir(destination)

""" FUNCTIONS """

def extract_param_vals(readLines, index):
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
    pd.DataFrame(params_dict).set_index(params_dict['index'])


""" EXECUTION """
params_dataframe(fnames, 'month')
