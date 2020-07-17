""" Outputs plots of all the analysis features for each of the six models.

Run this script from the bash shell. """

import sys
sys.path.append("./../../../core/")

import os
import inv_flux
import pickle
import matplotlib.pyplot as plt


def main():
    input_file = sys.argv[1]
    model_name = sys.argv[2]

    input_dataset = pickle.load(open(input_file, "rb"))

    df = inv_flux.Analysis(input_dataset)

    """Parameters."""
    window_size = int(sys.argv[3]) # Usually 25.
    period = sys.argv[4]
    fc = 1/float(period) # Cut-off frequency for bandpass func.
    btype = "low"
    deseasonalise_first = True # Tune to false if deseasonalisation not wanted in bandpass func.

    if input_file.endswith("year.pik"):
        fs = 1
        output_folder = f"./../../../../output/inversions/analysis/year/{model_name}/"
    elif input_file.endswith("spatial.pik"):
        fs = 12
        output_folder = f"./../../../../output/inversions/analysis/monthly/{model_name}/"
        window_size *= 12
    else:
        raise TypeError("Input file must end in either year or spatial.")


    """ Plots"""
    variables = ["Earth_Land", "South_Land", "North_Land", "Tropical_Land",
                "Earth_Ocean", "South_Ocean", "North_Ocean", "Tropical_Ocean"]

    for variable in variables:
        plt.clf()
        cascading_window_trend(variable, df, output_folder, window_size)
        plt.clf()
        psd(variable, df, fs, output_folder)

        if input_file.endswith("spatial.pik"):
            deseasonalise(variable, df, output_folder)
            bandpass(variable, df, fc, fs, btype, deseasonalise_first, output_folder, period)


def cascading_window_trend(variable, df, output_folder, window_size):

    roll_df, r_df = df.cascading_window_trend(variable, window_size, True, True)

    plt.savefig(f"{output_folder}{variable}/cascading_window_{str(window_size)}_{variable}.png")

    pickle.dump(roll_df, open(f"{output_folder}{variable}/cascading_window_{str(window_size)}_{variable}.pik", "wb"))
    pickle.dump(r_df, open(f"{output_folder}{variable}/cascading_window_pearson_{str(window_size)}_{variable}.pik", "wb"))


def psd(variable, df, fs, output_folder):

    psd = df.psd(variable, fs, plot=True)
    plt.savefig(f"{output_folder}{variable}/psd_{variable}.png")
    pickle.dump(psd, open(f"{output_folder}{variable}/psd_{variable}.pik", "wb"))


def deseasonalise(variable, df, output_folder):
    deseason = df.deseasonalise(variable)
    pickle.dump(deseason, open(f"{output_folder}{variable}/deseasonalise_{variable}.pik", "wb"))


def bandpass(variable, df, fc, fs, btype, deseasonalise_first, output_folder, period):
    bandpass = df.bandpass(variable, fc, fs, btype=btype, deseasonalise_first=deseasonalise_first)
    if deseasonalise_first:
        bandpass_fname = f"{output_folder}{variable}/bandpass_{period}_{btype}_{variable}_deseason.pik"
    else:
        bandpass_fname = f"{output_folder}{variable}/bandpass_{period}_{btype}_{variable}.pik"
    pickle.dump(bandpass, open(bandpass_fname, "wb"))



if __name__ == "__main__":
    main()
