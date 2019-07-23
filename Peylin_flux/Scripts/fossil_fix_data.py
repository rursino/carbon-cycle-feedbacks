""" Obtains arrays of land and ocean fluxes from models with fossil fuels removed in all models except for Rayner (as Rayner does not have fossil included in its flux values.
The arrays are then serialised as binary pickle files and stored in data_fixed directory.

Run this script from the bash shell. """

import xarray as xr
import pickle
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    output = sys.argv[2]
    
    df = xr.open_dataset(input_file)
    var = list(df.var())
    
    fossil = df[var[0]]
    land = df[var[1]]
    ocean = df[var[2]]
    
    land_fix = land-fossil
    ocean_fix = ocean-fossil

    dataset_fix = xr.Dataset({'Terrestrial_flux': land_fix, 'Ocean_flux': ocean_fix})
    
    pickle.dump(dataset_fix, open(output, "wb"))
    
    print("{} dataset successfully created and serialised.".format(output))
