""" Outputs plots of all the model evaluation features for each of the six models.

Run this script from the bash shell. """

import sys
sys.path.append("./../../../core/")

import os
import TRENDY_flux
import pickle
import xarray as xr
import matplotlib.pyplot as plt


def main():
    input_file = sys.argv[1]
    window_size = int(sys.argv[2])

    model_name = input_file.split("/")[-2]
    output_folder = ("./../../../../output/TRENDY/model_evaluation/"
                     f"{model_name}/")

    assert input_file.endswith("year.nc")

    ds = xr.open_dataset(input_file)
    df = TRENDY_flux.ModelEvaluation(ds)

    """ Plots."""
    TS_OUTPUT = f"{output_folder}regress_timeseries_to_GCP"
    REGRESS_OUTPUT = (f"{output_folder}"
                      f"regress_cascading_window{window_size}"
                      f"_trend_to_GCP")
    TREND_OUTPUT = f"{output_folder}compare_trend_to_GCP"


    df.regress_timeseries_to_GCP("both")
    plt.savefig(TS_OUTPUT + f".png")

    plt.clf()
    linreg = df.regress_cascading_window_trend_to_GCP(window_size, "time", "both")
    plt.savefig(REGRESS_OUTPUT + f"_time.png")
    pickle.dump(linreg, open(REGRESS_OUTPUT + f"_time.pik", "wb"))

    plt.clf()
    linreg = df.regress_cascading_window_trend_to_GCP(window_size, "CO2", "both")
    plt.savefig(REGRESS_OUTPUT + f"_CO2.png")
    pickle.dump(linreg, open(REGRESS_OUTPUT + f"_CO2.pik", "wb"))

    plt.clf()
    results = df.compare_trend_to_GCP()
    plt.savefig(TREND_OUTPUT + f".png")
    pickle.dump(results, open(TREND_OUTPUT + f".pik", "wb"))


if __name__ == "__main__":
    main()
