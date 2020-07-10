""" Outputs dataframes of yearly, decadal and whole time integrations for TRENDY
fluxes (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell.

"""

""" IMPORTS """
import sys
sys.path.append("./../../core/")
import TRENDY_flux as TRENDYf

import os
import xarray as xr
import pickle


""" SETUP """


""" FUNCTIONS """
def main(input_file, output_folder, ui=False):
    if ui:
        print("="*30)
        print("TRENDY")
        print("="*30 + '\n')
        print(f"Working with: {input_file}")
        print('-'*30)

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

    if os.path.isdir(output_folder):
        if ui:
            print("Directory %s already exists" % output_folder)
    else:
        os.mkdir(output_folder)

    # Output files after directory successfully created.
    for freq in arrays:
        destination = f"{output_folder}/{freq}.nc"
        try:
            arrays[freq].to_netcdf(destination)
        except:
            if ui:
                print(f": {freq.upper()}:fail")
        else:
            if ui:
                print(f": {freq.upper()}:pass")

    if ui:
        print("All files created!\n")


""" EXECUTION """
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    main(input_file, output_folder, True)
