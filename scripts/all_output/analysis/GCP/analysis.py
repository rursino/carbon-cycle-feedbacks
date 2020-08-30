""" Outputs plots of all the analysis features for each of the six models.

Run this script from the bash shell. """


""" IMPORTS """
import sys
import os
from core import GCP_flux
import pickle
import matplotlib.pyplot as plt
import xarray as xr
from tqdm import tqdm


""" INPUTS """

# Parameters
window_size = 10
period = 10

fc = 1/period # Cut-off frequency for bandpass func.
btype = "low"
deseasonalise_first = True # Tune to false if deseasonalisation
                           # not wanted in bandpass func.


""" FUNCTIONS """
# The following functions provide results and plots for each function in the
# Analysis class and saves them as files in the output main directory.

def cascading_window_trend(indep, variable):

    roll_df, r_df = df.cascading_window_trend(indep, window_size, True, True, True)

    CASCADING_OUTPUT_DIR = f"{output_folder}/window_size{str(window_size)}/"
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

    psd = df.psd(plot=True)
    plt.savefig(f"{output_folder}/psd_{variable}.png")
    pickle.dump(psd, open(f"{output_folder}/psd_{variable}.pik", "wb"))

def bandpass(variable):
    bandpass = df.bandpass(fc, btype=btype)

    BANDPASS_OUTPUT_DIR = f"{output_folder}/period{period}/"

    bandpass_fname = (BANDPASS_OUTPUT_DIR +
                      f"bandpass_{period}_{btype}_{variable}.pik")

    try:
        pickle.dump(bandpass, open(bandpass_fname, "wb"))
    except FileNotFoundError:
        os.mkdir(BANDPASS_OUTPUT_DIR)
        pickle.dump(bandpass, open(bandpass_fname, "wb"))


""" EXECUTION """
if __name__ == "__main__":

    variables = ["land sink", "land sink (model)", "ocean sink"]
    output_varnames = {
        "land sink": "land",
        "land sink (model)": "landmodel",
        "ocean sink": "ocean"
    }

    # Plots.
    for variable in tqdm(variables):
        output_varname = output_varnames[variable]
        df = GCP_flux.Analysis(variable)
        output_folder = f'./../../../../output/GCP/analysis/{output_varname}/'

        plt.close()
        cascading_window_trend('time', output_varname)
        plt.close()
        cascading_window_trend('CO2', output_varname)
        plt.close()
        psd(output_varname)

        bandpass(output_varname)
