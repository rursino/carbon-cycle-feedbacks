""" Outputs dataframes of yearly, decadal and whole time integrations for TRENDY
fluxes (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell.

"""

""" IMPORTS """
import sys
sys.path.append("./../../core/")
import TRENDY_flux as TRENDYf

from importlib import reload
reload(TRENDYf)

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

    df = (TRENDYf
            .SpatialAgg(data = input_file)
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


import pandas as pd
from datetime import datetime


df = TRENDYf.SpatialAgg('./../../../data/TRENDY/models/CABLE-POP/S1/CABLE-POP_S1_nbp.nc')
df.time_range()


df = df.data

time_resolution = "M"
tformat = "%Y-%m" if time_resolution == "M" else "%Y"

def format_time(time):

    return (pd
                .to_datetime(datetime.strptime(
                time.strftime(tformat), tformat)
                )
                .strftime(tformat)
            )

try:
    start_time = format_time(df.time.values[0])
except AttributeError:
    start_time = format_time(pd.to_datetime(df.time.values[0]))

try:
    end_time = format_time(df.time.values[-1])
except AttributeError:
    end_time = format_time(pd.to_datetime(df.time.values[-1]))
start_time
end_time

arg_time_range = pd.date_range(start=start_time, end=end_time,freq=time_resolution).strftime(tformat)

arg_time_range if False else slice(arg_time_range[0], arg_time_range[-1])
