""" Runs the linear_regression_time function from invf.Analaysis on all the 6 models with both land and ocean.
Run this script from the bash shell. """

import sys
sys.path.append("./../core/")

import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    variable = sys.argv[2]
    time_dict = sys.argv[3]
    plot_file = sys.argv[4]
    output_file = sys.argv[5]
    
    if input_file.endswith(".pik"):
        input_file = pickle.load(open(input_file, 'rb'))

    if time_dict.endswith(".pik"):
        time_dict = pickle.load(open(time_dict, 'rb'))
    
    df = inv_flux.Analysis(data=input_file)
    df = inv_flux.Analysis(df.cftime_to_datetime())
    
    regression = df.linear_regression_time(time=time_dict, variable=variable, save_plot=plot_file)
    print(f"Plot saved to {plot_file}")
    
    pickle.dump(regression, open(output_file, 'wb'))
    print(f"Regression data saved to {output_file}")
