""" Outputs plots of all the model evaluation features for each of the six models.

Run this script from the bash shell. """

import sys
sys.path.append("./../core/")

import os
import inv_flux
import pickle
import matplotlib.pyplot as plt

def main():
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    assert input_file.endswith("year.pik")
    
    input_file = pickle.load(open(input_file, "rb"))
    df = inv_flux.ModelEvaluation(input_file)
    
    # LAND
    df.plot_vs_GCP("land", "time")
    plt.savefig(f"{output_folder}plot_vs_GCP_land.png")
    
    plt.clf()
    linreg = df.regress_timeseries_to_GCP("land", True)
    plt.savefig(f"{output_folder}regress_timeseries_to_GCP_land.png")
    pickle.dump(linreg, open(f"{output_folder}regress_timeseries_to_GCP_land.pik", "wb"))
    
    plt.clf()
    results = df.compare_trend_to_GCP("land")
    plt.savefig(f"{output_folder}compare_trend_to_GCP_land.png")
    pickle.dump(results, open(f"{output_folder}compare_trend_to_GCP_land.pik", "wb"))
    
    # OCEAN
    df.plot_vs_GCP("ocean", "time")
    plt.savefig(f"{output_folder}plot_vs_GCP_ocean.png")
    
    plt.clf()
    linreg = df.regress_timeseries_to_GCP("ocean", True)
    plt.savefig(f"{output_folder}regress_timeseries_to_GCP_ocean.png")
    pickle.dump(linreg, open(f"{output_folder}regress_timeseries_to_GCP_ocean.pik", "wb"))
    
    plt.clf()
    results = df.compare_trend_to_GCP("ocean")
    plt.savefig(f"{output_folder}compare_trend_to_GCP_ocean.png")
    pickle.dump(results, open(f"{output_folder}compare_trend_to_GCP_ocean.pik", "wb"))
    

if __name__ == "__main__":
    main()
