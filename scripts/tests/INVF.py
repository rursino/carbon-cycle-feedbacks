""" Test the accuracy of the output from the execution of the
INVF.spatial_integration function.
"""

""" IMPORTS """
import sys
sys.path.append("./../core/")
import inv_flux as invf
import numpy as np

from importlib import reload
reload(invf)


""" INPUTS """

fJENA = "./../../data/inversions/fco2_JENA-s76-4-2-2018_June2018-ext3_1976-2017\
_monthlymean_XYT.nc"
JENA = invf.SpatialAgg(fJENA)

sp = JENA.spatial_integration()
sp
