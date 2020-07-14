""" Output of results from FeedbackAnalysis.
"""

""" IMPORTS """
import sys
sys.path.append("./core")
import FeedbackAnalysis as FA


""" INPUTS """
DIR = './../../../'
OUTPUT_DIR = DIR + 'output/'

uptake = ('inversions', 'CAMS')
temp = 'CRUTEM'
time = 'year'
CO2 = 'raw'
sink = 'Earth_Land'
time_range = None

fname = f'feedbacks/{uptake[1]}_{temp}_{time}_{CO2}_{sink}_{time_range}.txt'


""" EXECUTION """
df = FA.FeedbackAnalysis(uptake=uptake, temp=temp, time=time, CO2=CO2,
                        sink=sink, time_range=time_range)
df.feedback_output(OUTPUT_DIR + fname)
