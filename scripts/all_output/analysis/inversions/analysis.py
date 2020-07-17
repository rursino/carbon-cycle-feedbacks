""" Outputs plots of all the analysis features for each of the six models.

Run this script from the bash shell. """


""" IMPORTS """
import sys
sys.path.append("./../../../core/")

import os
import inv_flux
import pickle
import matplotlib.pyplot as plt
import xarray as xr

from importlib import reload
reload(inv_flux)


""" INPUTS """
input_file = "./../../../../output/inversions/spatial/output_all/CAMS/month.nc"#sys.argv[1]
model_name = input_file.split("/")[-2]

# Parameters
window_size = 25#int(sys.argv[2]) # Usually 25.
period = 10#sys.argv[3]

fc = 1/float(period) # Cut-off frequency for bandpass func.
btype = "low"
deseasonalise_first = True # Tune to false if deseasonalisation not wanted in bandpass func.

OUTPUT_DIR = "./../../../../output/"
output_folder = OUTPUT_DIR + f"inversions/analysis/year/{model_name}/"


""" FUNCTIONS """
def cascading_window_trend(variable):

    roll_df, r_df = df.cascading_window_trend(variable, window_size, True, True)

    return roll_df, r_df

    plt.savefig(f"{output_folder}{variable}/cascading_window_{str(window_size)}_{variable}.png")

    pickle.dump(roll_df, open(f"{output_folder}{variable}/cascading_window_{str(window_size)}_{variable}.pik", "wb"))
    pickle.dump(r_df, open(f"{output_folder}{variable}/cascading_window_pearson_{str(window_size)}_{variable}.pik", "wb"))

def psd(variable):

    psd = df.psd(variable, fs, plot=True)
    plt.savefig(f"{output_folder}{variable}/psd_{variable}.png")
    pickle.dump(psd, open(f"{output_folder}{variable}/psd_{variable}.pik", "wb"))


def deseasonalise(variable):
    deseason = df.deseasonalise(variable)
    pickle.dump(deseason, open(f"{output_folder}{variable}/deseasonalise_{variable}.pik", "wb"))


def bandpass(variable):
    bandpass = df.bandpass(variable, fc, fs, btype=btype, deseasonalise_first=deseasonalise_first)
    if deseasonalise_first:
        bandpass_fname = f"{output_folder}{variable}/bandpass_{period}_{btype}_{variable}_deseasonal.pik"
    else:
        bandpass_fname = f"{output_folder}{variable}/bandpass_{period}_{btype}_{variable}.pik"
    pickle.dump(bandpass, open(bandpass_fname, "wb"))


""" EXECUTION """
if __name__ == "__main__":

    ds = xr.open_dataset(input_file)
    df = inv_flux.Analysis(ds)

    if input_file.endswith("year.nc"):
        fs = 1
    elif input_file.endswith("month.nc"):
        fs = 12
        window_size *= 12
    else:
        raise TypeError("Input file must end in either year or month.")

    # Plots
    variables = ["Earth_Land", "South_Land", "North_Land", "Tropical_Land",
                "Earth_Ocean", "South_Ocean", "North_Ocean", "Tropical_Ocean"]

    for variable in variables:
        plt.clf()
        cascading_window_trend(variable)
        plt.clf()
        psd(variable)

        if input_file.endswith("month.nc"):
            deseasonalise(variable)
            bandpass(variable)
