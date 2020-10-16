# Signs reversed from CMIP to suit sign in thesis.
# Units in CMIP: beta = GtC/ppm, gamma = GtC/K
# Adjustments here:
    # beta /= 2.12 to get GtC/GtC = 1
    # dividing by 140 years to get units /yr

C4MIP = {
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

whole_CMIP = {}
for param in C4MIP:
    whole_CMIP[param] = [C4MIP[param], CMIP5[param], CMIP6[param]]

import pandas as pd

df = pd.DataFrame(whole_CMIP, index=['C4MIP', 'CMIP5', 'CMIP6'])

def get_dataframe(df, length_of_cmip_period=140):
    phi = 0.015 / 2.12
    rho = 1.93

    df['u_gamma_land'] = df['gamma_land'] * phi / rho
    df['u_gamma_ocean'] = df['gamma_ocean'] * phi / rho

    df['alpha_land'] = df['beta_land'] + df['u_gamma_land']
    df['alpha_ocean'] = df['beta_ocean'] + df['u_gamma_ocean']

    df = df[['beta_land', 'gamma_land', 'u_gamma_land', 'alpha_land',
             'beta_ocean', 'gamma_ocean', 'u_gamma_ocean', 'alpha_ocean']]

    df = df / length_of_cmip_period

    return df

final = get_dataframe(df)

final

caption = ("Values of the feedback parameters from the C4MIP, CMIP5 and CMIP6"
"intercomparisons, as well as a conversion to $u_{\gamma}$ using the values of "
"$\gamma$ in each intercomparison. Note that the values of the parameters have "
"been converted to match the units used in this study, and the signs have been "
"reversed.")
latex_code = final.to_latex(column_format='|c|cccc|cccc|',
                            label='tab:cmip_fb',
                            caption=caption,
                            float_format="%.3f")
latex_code = latex_code.replace("toprule", "hline")
latex_code = latex_code.replace("midrule", "hline")
latex_code = latex_code.replace("bottomrule", "hline")
print(latex_code)
