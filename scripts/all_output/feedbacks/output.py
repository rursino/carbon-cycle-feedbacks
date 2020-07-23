""" Output of results from FeedbackAnalysis.
"""

""" IMPORTS """
from itertools import *
import json
from tqdm import tqdm

import sys
sys.path.append("./../../core/")
import FeedbackAnalysis as FA

from importlib import reload
reload(FA)


""" INPUTS """
timerange_inputs = json.load(open("timerange_inputs.json", "r"))

inversions = list(product(['inversions'],
                          ['CAMS', 'Rayner', 'CTRACKER',
                           'JAMSTEC', 'JENA_s76', 'JENA_s85'])
                 )

TRENDYs_product = list(product(['TRENDY'],
                               ['CABLE-POP', 'CLASS-CTEM', 'JSBACH',
                                'LPJ-GUESS', 'OCN'],
                               ['S1', 'S3'],
                               ['nbp']
                              )
                      )
TRENDYs = [(i[0], (f'{i[1]}_{i[2]}_{i[3]}')) for i in TRENDYs_product]

regions = ["Earth", "South", "North", "Tropical"]

tempsink_invs = list((input[0][0], input[1]+input[0][1]) for
                input in product(zip(['CRUTEM', 'HadSST'], ['_Land', '_Ocean']),
                                 regions))
tempsink_TRENDYs = list(product(['CRUTEM'],
                                (region + '_Land' for region in regions)
                               )
                       )

time = ['year', 'month']

# inputs = (list(product(inversions, tempsink_invs, time)) +
#          list(product(TRENDYs, tempsink_TRENDYs, time))
#          )
inputs = list(product(TRENDYs, tempsink_TRENDYs, time))

del u
u = inputs[1]
u

uptake, tempsink, time = u
temp, sink = tempsink
uptake, temp, sink, time
time_start, time_stop = timerange_inputs[uptake[0]][uptake[1]]
time_start = int(time_start)
time_stop = int(time_stop)
time_range = build_time_ranges(time_start, time_stop)[1]

if time == "month":
    time_range = (f'{time_range[0]}-01', f'{time_range[1]}-12')
slice_time_range = slice(*time_range)
slice_time_range
# print(time_start, time_stop, time, slice_time_range)

df = FA.FeedbackAnalysis(uptake=uptake, temp=temp, time=time, sink=sink,
                         time_range=slice_time_range)
df.feedback_output()


""" FUNCTIONS """
def build_time_ranges(time_start, time_stop):
    # Handle years before 1959, where CO2 data starts, because years before
    # 1959 don't contain available data for analysis.
    time_start = max(1959, time_start)

    time_ranges = []
    current_time = time_start

    while current_time < time_stop:
        time_ranges.append((str(current_time), str(current_time + 10)))
        current_time += 10

    if current_time - time_stop > 0:
        time_ranges.pop(-1)
        time_ranges.append((str(time_stop - 10), str(time_stop)))

    return time_ranges
inputs

""" EXECUTION """
for input in tqdm(inputs):
    uptake, tempsink, time = input
    temp, sink = tempsink

    time_start, time_stop = timerange_inputs[uptake[0]][uptake[1]]
    time_start = int(time_start)
    time_stop = int(time_stop)
    time_ranges = build_time_ranges(time_start, time_stop)

    for time_range in time_ranges:
        if time == "month":
            time_range = (f'{time_range[0]}-01', f'{time_range[1]}-12')
        slice_time_range = slice(*time_range)

        df = FA.FeedbackAnalysis(uptake=uptake, temp=temp, time=time, sink=sink,
                                 time_range=slice_time_range)
        df.feedback_output()
