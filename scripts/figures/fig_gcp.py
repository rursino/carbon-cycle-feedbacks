""" IMPORTS """
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

import sys
sys.path.append('./../core/')
import GCP_flux as GCPf


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"

land_GCPf = GCPf.Analysis("land sink")
ocean_GCPf = GCPf.Analysis("ocean sink")
land = land_GCPf.data
ocean = ocean_GCPf.data

df = land_GCPf.all_data

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]


""" FIGURES """
def gcp_landocean(save=False):
    sns.set_style("darkgrid")

    plt.figure(figsize=(15,8))
    plt.plot(land, color='g')
    plt.plot(ocean, color='b')

    plt.title("GCP Uptake: 1959 - 2018", fontsize=32)
    plt.xlabel("Year", fontsize=24)
    plt.ylabel("Uptake flux to the atmosphere (GtC/yr)", fontsize=16)

    plt.legend(["land", "ocean"], fontsize=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_landocean.png")

def gcp_simple_regression(save=False):
    sns.set_style("darkgrid")

    zip_list = zip(['211', '212'], ['land', 'ocean'], [land, ocean])

    plt.figure(figsize=(15,8))
    for subplot, var, sink in zip_list:
        plt.subplot(subplot)

        slope, intercept, rvalue = stats.linregress(co2, sink)[:3]
        plt.plot(co2, sink)
        plt.plot(co2, co2*slope + intercept)

        text = f"Slope: {slope:.2f} GtC/ppm$^2$\nr = {rvalue:.2f}"
        xtext = co2.min() + 0.8 * (co2.max() - co2.min())
        ytext = sink.min() +  0.8 * (sink.max() - sink.min())
        plt.text(xtext, ytext, text, fontsize=15)

    plt.title("GCP Uptake (with regression)", fontsize=32)
    plt.xlabel("CO2 concentrations (ppm)", fontsize=16)
    plt.ylabel("Uptake flux to the atmosphere (GtC/ppm)", fontsize=16)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_simple_regression.png")


gcp_simple_regression(save=False)
