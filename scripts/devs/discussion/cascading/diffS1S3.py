""" Difference between S1 and S3 in cascading window trend.
"""

""" IMPORTS """
import pickle
import matplotlib.pyplot as plt

""" INPUTS """
INPUT_DIR = './../../../../output/TRENDY/analysis/year/'
fS1 = INPUT_DIR + 'JSBACH_S1_nbp/Earth_Land/window_size10/cascading_window_10_Earth_Land_CO2.pik'
fS3 = INPUT_DIR + 'JSBACH_S3_nbp/Earth_Land/window_size10/cascading_window_10_Earth_Land_CO2.pik'

""" DEVS """
S1 = pickle.load(open(fS1, 'rb'))['10-year trend slope']
S3 = pickle.load(open(fS3, 'rb'))['10-year trend slope']

plt.plot(S1)
plt.plot(S3)
