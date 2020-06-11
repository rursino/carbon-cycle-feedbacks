""" Test the accuracy of the output from the execution of the
TRENDY.spatial_integration function.
"""

""" IMPORTS """
import sys
sys.path.append("./../core/")
import TRENDY_flux as Tflux
import numpy as np

from importlib import reload
reload(Tflux)


""" INPUTS """
dir = "./../../data/TRENDY/models/LPJ-GUESS/S0/"
fname = dir + "LPJ-GUESS_S0_cVeg.nc"

df = Tflux.SpatialAgg(fname)

df.spatial_integration()
