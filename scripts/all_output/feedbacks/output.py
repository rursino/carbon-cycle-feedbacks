""" Output of results from FeedbackAnalysis.
"""

""" IMPORTS """
from itertools import *
import json

import sys
sys.path.append("./../../core/")
import FeedbackAnalysis as FA

# from importlib import reload
# reload(FA)


""" INPUTS """
timerange_inputs = json.load(open("timerange_inputs.json", "r"))

inversions = list(product(['inversions'],
                          ['CAMS', 'Rayner', 'CTRACKER',
                           'JAMSTEC', 'JENA_s76', 'JENA_s85'])
                 )

TRENDYs = list(product(['TRENDY'],
                       ['CABLE-POP', 'CLASS-CTEM', 'JSBACH', 'LPJ-GUESS', 'OCN'],
                       ['S1', 'S3'],
                       ['nbp']
                      )
              )

temp_sink = zip(['CRUTEM', 'HadSST'], ['_Land', '_Ocean'])
regions = ["Earth", "South", "North", "Tropical"]
tempsink = list((input[0][0], input[1]+input[0][1]) for
                input in product(temp_sink, regions))

time = ['year', 'month']

inputs = list(product(inversions, tempsink, time))


""" FUNCTIONS """
def build_time_ranges(time_start, time_stop):
    time_ranges = []
    current_time = time_start

    while current_time < time_stop:
        time_ranges.append((str(current_time), str(current_time + 10)))
        current_time += 10

    if current_time - time_stop > 0:
        time_ranges.pop(-1)
        time_ranges.append((str(time_stop - 10), str(time_stop)))

    return time_ranges


""" EXECUTION """
for input in inputs:
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
