""" Outputs plots of all the analysis features for each of the TRENDY models.

Run this script from the bash shell. """


""" IMPORTS """
import sys
sys.path.append("./../../../core/")

import os
import TRENDY_flux
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm


""" INPUTS """
input_file = sys.argv[1]
model_name = input_file.split("/")[-2]
timeres = input_file.split("/")[-1].split(".")[0]

# Parameters
window_size = int(sys.argv[2])
period = sys.argv[3]

fc = 1/float(period) # Cut-off frequency for bandpass func.
btype = "low"
deseasonalise_first = True # Tune to false if deseasonalisation
                           # not wanted in bandpass func.

output_folder = ('./../../../../output/TRENDY/analysis/'
                 f'{timeres}/{model_name}/')


""" FUNCTIONS """
# The following functions provide results and plots for each function in the
# Analysis class and saves them as files in the output main directory.

def cascading_window_trend(indep, variable):

    roll_df, r_df = df.cascading_window_trend(indep, variable, window_size,
                                              True, True, True)

    CASCADING_OUTPUT_DIR = f"{output_folder}{variable}/window_size{str(window_size)}/"
    try:
        plt.savefig(CASCADING_OUTPUT_DIR +
                    f"cascading_window_{str(window_size)}_{variable}_{indep}.png")
    except FileNotFoundError:
        os.mkdir(CASCADING_OUTPUT_DIR)
        plt.savefig(CASCADING_OUTPUT_DIR +
                    f"cascading_window_{str(window_size)}_{variable}_{indep}.png")

    pickle.dump(roll_df, open(CASCADING_OUTPUT_DIR +
            f"cascading_window_{str(window_size)}_{variable}_{indep}.pik", "wb"))
    pickle.dump(r_df, open(CASCADING_OUTPUT_DIR +
        f"cascading_window_pearson_{str(window_size)}_{variable}_{indep}.pik", "wb"))

def psd(variable):

    psd = df.psd(variable, fs, plot=True)
    plt.savefig(f"{output_folder}{variable}/psd_{variable}.png")
    pickle.dump(psd, open(f"{output_folder}{variable}/psd_{variable}.pik", "wb"))

def deseasonalise(variable):
    deseason = df.deseasonalise(variable)
    pickle.dump(deseason, open(f"{output_folder}{variable}/"
                               f"deseasonalise_{variable}.pik", "wb"))

def bandpass(variable):
    bandpass = df.bandpass(variable, fc, fs, btype=btype,
                           deseasonalise_first=deseasonalise_first)

    BANDPASS_OUTPUT_DIR = f"{output_folder}{variable}/period{period}/"

    if deseasonalise_first:
        bandpass_fname = (BANDPASS_OUTPUT_DIR +
                          f"bandpass_{period}_{btype}_{variable}_deseasonal.pik")
    else:
        bandpass_fname = (BANDPASS_OUTPUT_DIR +
                          f"bandpass_{period}_{btype}_{variable}.pik")

    try:
        pickle.dump(bandpass, open(bandpass_fname, "wb"))
    except FileNotFoundError:
        os.mkdir(BANDPASS_OUTPUT_DIR)
        pickle.dump(bandpass, open(bandpass_fname, "wb"))


""" EXECUTION """
if __name__ == "__main__":

    ds = xr.open_dataset(input_file)
    df = TRENDY_flux.Analysis(ds)

    if timeres == "year":
        fs = 1
    elif timeres == "month":
        fs = 12

    # Plots
    variables = ["Earth_Land", "South_Land", "North_Land", "Tropical_Land"]

    for variable in tqdm(variables):
        plt.close()
        cascading_window_trend('time', variable)
        plt.close()
        cascading_window_trend('CO2', variable)
        plt.close()
        psd(variable)

        if timeres == "month":
            deseasonalise(variable)
            bandpass(variable)
