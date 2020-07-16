""" Output of results from FeedbackAnalysis.
"""

""" IMPORTS """
from itertools import *

import sys
sys.path.append("./../../core/")
import FeedbackAnalysis as FA

from importlib import reload
reload(FA)


""" INPUTS """
inversions = list(product(['inversions'],
                          ['CAMS', 'Rayner', 'CTRACKER',
                           'JAMSTEC', 'JENA_s76', 'JENA_s85']
                         )
                 )
TRENDYs = list(product(['TRENDY'],
                       ['CABLE-POP', 'CLASS-CTEM', 'JSBACH', 'LPJ-GUESS', 'OCN'],
                       ['S1', 'S3'],
                       ['nbp']
                      )
              )
uptake = inversions + TRENDYs

temp = ['CRUTEM']

# timeCO2 = [('year', 'raw'),
#            ('month', 'weighted'),
#            ('month', 'deseasonal')
#           ]
time, CO2 = ['year'], ['raw']

sink = ['Earth_Land']

time_range = None

all_inputs = product(temp, time, CO2, sink)
inputs = []
for uptake_index in uptake:
    input = []
    for index in uptake_index:
        input.append(index)
    for index in all_inputs:
        input.append(index)
    inputs.append(index)
inputs

""" EXECUTION """
df = FA.FeedbackAnalysis(uptake=uptake, temp=temp, time=time, CO2=CO2,
                        sink=sink, time_range=time_range)
df.feedback_output()
