""" Obtains arrays of land and ocean fluxes from models with fossil fuels removed in all models except for Rayner (as Rayner does not have fossil included in its flux values.
The arrays are then serialised as binary pickle files and stored in data_fixed directory.

Run this script from the bash shell. """

import xarray as xr
import pickle
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    land_output = sys.argv[2]
    ocean_output = sys.argv[3]
    
    df = xr.open_dataset(input_file)
    var = list(df.var())
    
    fossil = df[var[0]]
    land = df[var[1]]
    ocean = df[var[2]]

    land_fix = land-fossil
    ocean_fix = ocean-fossil
    
    pickle.dump(land_fix, open(land_output, "wb"))
    pickle.dump(ocean_fix, open(ocean_output, "wb"))
    
    print("{} and {} successfully created and serialised.".format(land_output, ocean_output))
