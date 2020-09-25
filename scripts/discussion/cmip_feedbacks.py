# Signs reversed from CMIP to suit sign in thesis.
# Units in CMIP: beta = GtC/ppm, gamma = GtC/K
# Adjustments here: beta /= 2.12 to get GtC/GtC = 1
# Adjustments not made here: getting /yr, not trivial.

CMIP4 = {
    'beta_land': -1.35 / 2.12, 'beta_ocean': -1.13 / 2.12,
    'gamma_land': 79, 'gamma_ocean': 30
}

CMIP6 = {
    'beta_land': -1.2 / 2.12, 'beta_ocean': -0.79 / 2.12,
    'gamma_land': 63.8, 'gamma_ocean': 17.2
}

CMIP5 = {
    'beta_land': -1.2 / 2.12, 'beta_ocean': -0.82 / 2.12,
    'gamma_land': 75.4, 'gamma_ocean': 17.3
}
