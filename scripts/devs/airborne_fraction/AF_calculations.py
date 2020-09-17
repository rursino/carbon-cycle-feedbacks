""" First developments of the calculation of the airborne fraction (AF) using
the feedback parameters and the equation used in Gregory2009 paper.
"""

""" IMPORTS """
import sys
sys.path.append('./../../core')
import FeedbackAnalysis as FA

import numpy as np
import matplotlib.pyplot as plt


""" PHI """
def phi(C):
    return np.log(C/290) * 3.74 / (np.log(2) * (C - 290))

phi(270)

x = np.linspace(280, 840)
y = phi(x)
plt.plot(x,y)
plt.ylim([0,0.3])


""" FUNCTIONS """
def TRENDY(variable, timeres):
    df = FA.FeedbackOutput("TRENDY", variable, timeres)

    param_dict = {}
    for param in ["beta", "gamma"]:
        merge = df.merge_params("S3", param)
        param_dict[param] = merge.mean(axis=1), merge.std(axis=1)

    # equation: 1/(1+beta+gamma_dimensionless)

    # Calculate dimensionless gamma = phi*gamma/rho
    # rho = time-dependent climate feedback factor, can make it constant i.e.
    # around 1.3 according to some AOGCMs.
    # phi = see phi func, but around 0.015

    beta = param_dict['beta'][0]
    gamma = param_dict['gamma'][0] * 0.015 / 1.3

    # Calculate airborne fraction as af.
    return 1/(1 + beta + gamma)

def inversions(variable, timeres):
    df = FA.FeedbackOutput("inversions", variable, timeres)

    # equation: 1/(1+beta+gamma_dimensionless)

    # Calculate dimensionless gamma = phi*gamma/rho
    # rho = time-dependent climate feedback factor, can make it constant i.e.
    # around 1.3 according to some AOGCMs.
    # phi = see phi func, but around 0.015

    beta = df.params_data['JENA_s76']['beta_median']
    gamma = df.params_data['JENA_s76']['gamma_median'] * 0.015 / 1.3

    # Calculate airborne fraction as af.
    return 1/(1 + beta + gamma)


""" EXECUTION """
TRENDY("Earth_Land", "month") / 2

1 / (1 / inversions("Earth_Land", "month") + 1 / inversions("Earth_Ocean", "month"))
