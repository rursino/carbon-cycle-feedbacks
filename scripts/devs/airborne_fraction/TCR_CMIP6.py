""" Implementation of TCR CMIP6 data to use for calculation of u_gamma and
ultimately the airborne fraction (and also for the feedback eqns).
"""


import numpy as np

TCR_CMIP6 = np.array([2.1, 2.0, 2.0, 1.7, 1.8, 1.7, 2.0, 2.0, 2.1, 2.5,
1.9, 2.7, 3.0, 2.6, 2.1, 2.1, 1.6, 1.8, 1.9, 1.7, 2.6, 2.6, 1.7, 1.3, 2.3,
1.4, 1.9, 1.6, 1.6, 1.7, 1.8, 1.6, 2.7, 1.6, 1.5, 2.3, 2.8])

TCR_CMIP6.mean()
TCR_CMIP6.std()


rho = 3.74 / TCR_CMIP6

rho.mean()
rho.std()

ata = 0.015 / rho

ata.mean()
ata.std()

3.74 / 2.01
