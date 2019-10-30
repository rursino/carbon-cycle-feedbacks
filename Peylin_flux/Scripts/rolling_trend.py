""" Runs the rolling_trend function from invf.Analaysis on all the 6 models with both land and ocean.
Run this script from the bash shell. """

import sys
import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    variable = sys.argv[2]
    plot_file = sys.argv[3]
    output_file = sys.argv[4]
    window_size = 10 #OR sys.argv[5] for input
    
    if input_file.endswith(".pik"):
        input_file = pickle.load(open(input_file, "rb"))
    
    df = inv_flux.Analysis(data=input_file)
    
    roll_df = df.rolling_trend(variable=variable, window_size=window_size, save_plot=plot_file)
    print(f"Plot saved to {plot_file}")
    
    pickle.dump(roll_df, open(output_file, 'wb'))
    print(f"Rolling trend results saved to {output_file}")
