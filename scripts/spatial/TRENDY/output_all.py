""" Outputs dataframes of yearly, decadal and whole time integrations for TRENDY
fluxes (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell.

"""

import sys
sys.path.append("./../../core/")
import TRENDY_flux as TRENDYf

import os
import xarray as xr
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    ds = xr.open_dataset(input_file)
    df = (TRENDYf
            .SpatialAgg(data = ds)
            .latitudinal_splits()
         )

    arrays = {
        "year": df,
        "decade": df.resample({'time': '10Y'}).sum(),
        "whole": df.sum()
    }

    print("="*30)
    print("TRENDY")
    print("="*30)

    if os.path.isdir(output_folder):
        print("Directory %s already exists" % output_folder)
    else:
        os.mkdir(output_folder)
        print("Successfully created the directory %s " % output_folder)

    # Output files after directory successfully created.
    for freq in arrays:
        array = arrays[freq]
        pickle.dump(array, open(f"{output_folder}/{freq}.pik", "wb"))
        print(f"Successfully created {output_folder}/{freq}.pik")

    print("All files created!\n")
