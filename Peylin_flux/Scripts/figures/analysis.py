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
    if input_file.endswith("year.pik"):
        fs = 1
        output_folder = f"./../../Output/analysis/year/{model_name}/"
        deseasonalise_first = False
    elif input_file.endswith("spatial.pik"):
        fs = 12
        output_folder = f"./../../Output/analysis/monthly/{model_name}/"
        deseasonalise_first = True # Tune to false if deseasonalisation not wanted in bandpass func.
    else:
        raise TypeError("Input file must end in either year or spatial.")

    window_size = sys.argv[3] # Usually 25.
    fc = 1/float(sys.argv[4]) # Cut-off frequency for bandpass func.
    btype = "low"
    
    
    """ Land plots."""
    roll_df, r_df = df.rolling_trend("Earth_Land", int(window_size), True, True)
    plt.savefig(f"{output_folder}rolling_trend_{window_size}_land.png")
    pickle.dump(roll_df, open(f"{output_folder}rolling_trend_{window_size}_land.pik", "wb"))
    pickle.dump(r_df, open(f"{output_folder}rolling_trend_pearson_{window_size}_land.pik", "wb"))
    
    plt.clf()
    psd = df.psd("Earth_Land", fs, plot=True)
    plt.savefig(f"{output_folder}psd_land.png")
    pickle.dump(psd, open(f"{output_folder}psd_land.pik", "wb"))
    
    if input_file.endswith("spatial.pik"):
        deseason = df.deseasonalise("Earth_Land")
        pickle.dump(deseason, open(f"{output_folder}deseasonalise_land.pik", "wb"))
    
    bandpass = df.bandpass("Earth_Land", fc, fs, btype=btype, deseasonalise_first=deseasonalise_first)
    if deseasonalise_first:
        bandpass_fname = f"{output_folder}bandpass_{sys.argv[4]}_{btype}_land_deseason.pik"
    else:
        bandpass_fname = f"{output_folder}bandpass_{sys.argv[4]}_{btype}_land.pik"
    pickle.dump(bandpass, open(bandpass_fname, "wb"))
    
    
    """ Ocean plots."""
    roll_df, r_df = df.rolling_trend("Earth_Ocean", int(window_size), True, True)
    plt.savefig(f"{output_folder}rolling_trend_{window_size}_ocean.png")
    pickle.dump(roll_df, open(f"{output_folder}rolling_trend_{window_size}_ocean.pik", "wb"))
    pickle.dump(r_df, open(f"{output_folder}rolling_trend_pearson_{window_size}_ocean.pik", "wb"))
    
    plt.clf()
    psd = df.psd("Earth_Ocean", fs, plot=True)
    plt.savefig(f"{output_folder}psd_ocean.png")
    pickle.dump(psd, open(f"{output_folder}psd_ocean.pik", "wb"))
    
    if input_file.endswith("spatial.pik"):
        deseason = df.deseasonalise("Earth_Ocean")
        pickle.dump(deseason, open(f"{output_folder}deseasonalise_ocean.pik", "wb"))
    
    bandpass = df.bandpass("Earth_Ocean", fc, fs, btype=btype, deseasonalise_first=deseasonalise_first)
    if deseasonalise_first:
        bandpass_fname = f"{output_folder}bandpass_{sys.argv[4]}_{btype}_ocean_deseason.pik"
    else:
        bandpass_fname = f"{output_folder}bandpass_{sys.argv[4]}_{btype}_ocean.pik"
    pickle.dump(bandpass, open(bandpass_fname, "wb"))



if __name__ == "__main__":
    main()
