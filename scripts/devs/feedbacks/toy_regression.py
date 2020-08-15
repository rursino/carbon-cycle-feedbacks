""" Create a toy set of data variables and perform regression for the sake of
learning and experimentation.
Refer to: https://www.hindawi.com/journals/jps/2014/673657/ for information on
Bayesian Inference
"""

""" IMPORTS """
import numpy as np
import pandas as pd
import statsmodels.api as sm


""" INPUTS """
X1 = np.array([1, 2, 4, 5, 6, 7, 8])
X2 = np.array([23, 26, 28, 30, 31, 34, 36])
Y = np.array([0, 0, 1, 2, 3, 4, 6])


""" EXECUTION """
df = pd.DataFrame(
    {
        'X1': X1,
        'X2': X2,
        'Y': Y
    }
)

X = sm.add_constant(df[['X1', 'X2']])

model = sm.OLS(Y, X).fit()
model.summary()
model.cov_params()
