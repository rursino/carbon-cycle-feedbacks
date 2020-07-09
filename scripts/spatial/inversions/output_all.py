""" Outputs dataframes of spatial, yearly, decadal and whole time integrations
(for all globe and regions) for each model. Output format is binary csv
through the use of pickle.

Run this script from the bash shell.
"""

import sys
sys.path.append("./../../core/")
import inv_flux

import os
import xarray as xr
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    ds = xr.open_dataset(input_file)
    df = (inv_flux
            .SpatialAgg(data = ds)
            .latitudinal_splits()
         )

    arrays = {
        "month": df,
        "year": df.resample({'time': 'Y'}).sum(),
        "decade": df.resample({'time': '10Y'}).sum(),
        "whole": df.sum()
    }

    print("="*30)
    print("INVERSIONS")
    print("="*30)

    if os.path.isdir(output_folder):
        print("Directory %s already exists" % output_folder)
    else:
        os.mkdir(output_folder)
        print("Successfully created the directory %s " % output_folder)

    # Output files after directory successfully created.
    for freq in arrays:
        destination = f"{output_folder}/{freq}.nc"
        arrays[freq].to_netcdf(destination)
        print(f"Successfully created {destination}")

    print("All files created!\n")
