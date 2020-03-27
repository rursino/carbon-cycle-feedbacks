""" Outputs plots of all the analysis features for each of the six models.

Run this script from the bash shell. """

import sys
sys.path.append("./../core/")

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
        output_folder = f"./../../Output/analysis/year/{model_name}/"
    elif input_file.endswith("spatial.pik"):
        fs = 12
        output_folder = f"./../../Output/analysis/monthly/{model_name}/"
        window_size *= 12
    else:
        raise TypeError("Input file must end in either year or spatial.")
    
    
    """ Global plots."""
    all_analyses("Earth_Land")
    all_analyses("Earth_Ocean")
    
    """ Regional plots."""
    

def rolling_trend(variable):
    
    roll_df, r_df = df.rolling_trend(variable, window_size, True, True)
    
    plt.savefig(f"{output_folder}rolling_trend_{str(window_size)}_{variable}.png")
    
    pickle.dump(roll_df, open(f"{output_folder}rolling_trend_{str(window_size)}_{variable}.pik", "wb"))
    pickle.dump(r_df, open(f"{output_folder}rolling_trend_pearson_{str(window_size)}_{variable}.pik", "wb"))
    

def psd(variable):
    
    psd = df.psd(variable, fs, plot=True)
    plt.savefig(f"{output_folder}psd_{variable}.png")
    pickle.dump(psd, open(f"{output_folder}psd_{variable}.pik", "wb"))

    
def deseasonalise(variable):
    deseason = df.deseasonalise(variable)
    pickle.dump(deseason, open(f"{output_folder}deseasonalise_{variable}.pik", "wb"))

    
def bandpass(variable):
    bandpass = df.bandpass(variable, fc, fs, btype=btype, deseasonalise_first=deseasonalise_first)
    if deseasonalise_first:
        bandpass_fname = f"{output_folder}bandpass_{period}_{btype}_{variable}_deseason.pik"
    else:
        bandpass_fname = f"{output_folder}bandpass_{period}_{btype}_{variable}.pik"
    pickle.dump(bandpass, open(bandpass_fname, "wb"))
    

def all_analyses(variable):
    plt.clf()
    rolling_trend(variable)
    plt.clf()
    psd(variable)
    deseasonalise(variable)
    bandpass(variable)


if __name__ == "__main__":
    main()
