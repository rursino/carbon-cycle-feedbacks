from scipy import stats
import numpy as np

n = 10
U = np.linspace(-9,-9-n-1,n)
dCedt = np.array([19*(1.02**i) for i in range(n)])
dCadt = dCedt + U
Ca = 300 + np.cumsum(dCadt)
Ca
A = dCadt / dCedt

stat = stats.linregress(Ca, U)
alpha = stat.slope
intercept = stat.intercept

alpha
A
1 / (1 - (1/0.02)*alpha)



pd.DataFrame({'U': U, 'dCedt': dCedt, 'dCadt': dCedt + U,
              'Ca': 300 + np.cumsum(dCadt), 'A': dCadt / dCedt})
