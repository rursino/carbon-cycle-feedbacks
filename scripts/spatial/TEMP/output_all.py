""" Outputs dataframes of monthly, yearly, decadal and whole time
temperature averages (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell.

"""

import sys
sys.path.append("./../../core/")
import TEMP

import os
import xarray as xr
import pickle

def main():
    ds = xr.open_dataset(input_file)
    df = (TEMP
            .SpatialAve(data = ds)
            .latitudinal_splits()
         )

    arrays = {
        "month": df,
        "year": df.resample({'time': 'Y'}).mean(),
        "decade": df.resample({'time': '10Y'}).mean(),
        "whole": df.mean()
    }

    print("="*30)
    print("TEMP")
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

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    ds = xr.open_dataset(input_file)
    df = (TEMP
            .SpatialAve(data = ds)
            .latitudinal_splits()
         )

    arrays = {
        "month": df,
        "year": df.resample({'time': 'Y'}).mean(),
        "decade": df.resample({'time': '10Y'}).mean(),
        "whole": df.mean()
    }

    print("="*30)
    print("TEMP")
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
