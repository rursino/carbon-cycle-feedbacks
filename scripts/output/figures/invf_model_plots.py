"""Python script (for the bash) to extract plots of carbon uptake timeseries of the six models and the GCP. Useful as a figure for the final thesis paper."""

import sys
sys.path.append("./../../core/")

import inv_flux as invf
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import pickle



def main():

    list_of_models = ["Rayner", "CAMS", "CTRACKER",
                      "JAMSTEC", "JENA_s76", "JENA_s85"]

    dict_of_models = {}
    for model in list_of_models:
        dict_of_models[model] = open_data(f"./../../../../output/inversions/raw/output_all/{model}_all/year.pik")

    dfGCP = pd.read_csv("./../../../data/GCP/budget.csv",
                        index_col=0,
                        usecols=[0,4,5,6],
                        skipfooter=1
                       )

    land = -dfGCP["land sink"] - dfGCP["budget imbalance"]
    ocean = -dfGCP["ocean sink"]

    plot(dict_of_models, land, "Earth_Land")
    plot(dict_of_models, ocean, "Earth_Ocean")


def open_data(fname):

    data = pickle.load(open(fname, "rb"))

    return invf.Analysis(data).cftime_to_datetime()


def plot(dict_of_models, GCP, var):

    plt.figure(figsize=(14,9))

    dict_of_models["Rayner"][var].plot()
    dict_of_models["CAMS"][var].plot()
    dict_of_models["CTRACKER"][var].plot()
    dict_of_models["JAMSTEC"][var].plot()
    dict_of_models["JENA_s76"][var].plot()
    dict_of_models["JENA_s85"][var].plot()

    if var == "Earth_Land":
        GCP_legend = "GCP Land"
        title = "Land Uptake"
    elif var == "Earth_Ocean":
        GCP_legend = "GCP Ocean"
        title = "Ocean Uptake"

    plt.plot(dict_of_models["JENA_s76"].time.values, GCP[17:], linewidth=7.0 , color='k')

    legend = ["Rayner", "CAMS", "CTRACKER", "JAMSTEC", "JENA_s76", "JENA_s85", GCP_legend]
    plt.legend(legend, fontsize=12)

    plt.title(title, fontsize=30)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('C flux to the atmosphere (GtC/yr)', fontsize=20)

    plt.show()



if "__name__" == "__main__":
    main()
