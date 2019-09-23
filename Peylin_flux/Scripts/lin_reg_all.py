""" Run this script from the bash shell. """

import sys
import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    variable = sys.argv[2]
    output_file = sys.argv[3]
    
    data = 