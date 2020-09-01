""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

import seaborn as sns
sns.set_style('darkgrid')

import sys
from copy import deepcopy
from core import GCP_flux as GCPf


""" INPUTS """
FIGURE_DIRECTORY = "./../../latex/thesis/figures/"

land_GCPf = GCPf.Analysis("land sink")
ocean_GCPf = GCPf.Analysis("ocean sink")
land = land_GCPf.data
ocean = ocean_GCPf.data

df = land_GCPf.all_data

co2 = pd.read_csv("./../../data/CO2/co2_year.csv").CO2[2:]


""" FUNCTIONS """
def bandpass_instance(instance, fc):
    bp_instance = deepcopy(instance)
    bp = instance.bandpass(fc)
    bp_instance.data = pd.Series(bp, index=bp_instance.data.index)

    return bp_instance


""" FIGURES """
def gcp_landocean(save=False):
    plt.figure(figsize=(15,8))
    plt.plot(land, color='g')
    plt.plot(ocean, color='b')

    plt.title("GCP Uptake: 1959 - 2018", fontsize=32)
    plt.xlabel("Year", fontsize=24)
    plt.ylabel("Uptake flux to the atmosphere (GtC/yr)", fontsize=16)

    plt.legend(["land", "ocean"], fontsize=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_landocean.png")

def gcp_simple_regression(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['land', 'ocean'], [land, ocean])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        regstats = stats.linregress(co2, sink)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(co2, sink)
        ax.plot(co2, co2*slope + intercept)
        text = f"Slope: {slope:.3f} GtC yr$^{'{-1}'}$ ppm$^{'{-1}'}$\nr = {rvalue:.3f}"
        xtext = co2.min() + 0.75 * (co2.max() - co2.min())
        ytext = sink.min() +  0.8 * (sink.max() - sink.min())
        ax.text(xtext, ytext, text, fontsize=15)

        stat_vals[var] = regstats

    axl.set_title("GCP Uptake (with regression)", fontsize=32, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel("Uptake flux to the atmosphere (GtC yr$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_simple_regression.png")

    if stat_values:
        return stat_vals

def gcp_cwt(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['land', 'ocean'], [land_GCPf, ocean_GCPf])

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        sink_cwt_df = sink.cascading_window_trend(indep="CO2", window_size=10)
        x = sink_cwt_df.index
        y = sink_cwt_df.values.squeeze()

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(x, y)
        ax.plot(x, x*slope + intercept)
        text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
        xtext = x.min() + 0.75 * (x.max() - x.min())
        ytext = y.min() +  0.8 * (y.max() - y.min())
        ax.text(xtext, ytext, text, fontsize=15)

        stat_vals[var] = regstats

    axl.set_title("Cascading Window 10-Year Trend", fontsize=32, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_cwt.png")

    if stat_values:
        return stat_vals

def gcp_powerspec(save=False):
    zip_list = zip(['211', '212'], ['land', 'ocean'], [land_GCPf, ocean_GCPf])

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        sink_psd = sink.psd()
        x = sink_psd.iloc[:,0]
        y = sink_psd.iloc[:,1]

        ax.semilogy(x, y)
        ax.invert_xaxis()

    axl.set_title("Power Spectrum: GCP Uptake", fontsize=32, pad=20)
    axl.set_xlabel(sink_psd.columns[0], fontsize=16, labelpad=10)
    axl.set_ylabel(sink_psd.columns[1], fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_powerspec.png")

def gcp_cwt_bandpass(save=False, stat_values=False, fc=1/10):
    zip_list = zip(
        ['211', '212'],
        ['land', 'ocean'],
        [bandpass_instance(land_GCPf, fc), bandpass_instance(ocean_GCPf, fc)]
            )

    stat_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        sink_cwt_df = sink.cascading_window_trend(indep="CO2", window_size=10)
        x = sink_cwt_df.index
        y = sink_cwt_df.values.squeeze()

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(x, y)
        ax.plot(x, x*slope + intercept)
        text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-1}'}$ ppm$^{'{-2}'}$\nr = {rvalue:.3f}"
        xtext = x.min() + 0.75 * (x.max() - x.min())
        ytext = y.min() +  0.8 * (y.max() - y.min())
        ax.text(xtext, ytext, text, fontsize=15)

        stat_vals[var] = regstats

    axl.set_title(f"Cascading Window 10-Year Trend\n(Low-pass: {1/fc} years)",
                  fontsize=24, pad=20)
    axl.set_xlabel("CO$_2$ concentrations (ppm)", fontsize=16, labelpad=10)
    axl.set_ylabel(r"$\alpha$   " + " (GtC yr$^{-1}$ ppm$^{-1}$)", fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_cwt_bandpass.png")

    if stat_values:
        return stat_vals


""" EXECUTION """
gcp_landocean(save=False)

stat_simple_regression = gcp_simple_regression(save=False, stat_values=True)
(stat_simple_regression['land'].slope * 60, stat_simple_regression['ocean'].slope * 60)

stat_cwt = gcp_cwt(save=False, stat_values=True)
(stat_cwt['land'].slope * 60 *(co2.iloc[-1] - co2.iloc[0])**2 / 1e3,
stat_cwt['ocean'].slope * 60 *(co2.iloc[-1] - co2.iloc[0])**2 / 1e3)

gcp_powerspec(save=False)

stat_cwt_bp = gcp_cwt_bandpass(save=False, stat_values=True, fc=1/25)
(stat_cwt_bp['land'].slope * 60 *(co2.iloc[-1] - co2.iloc[0])**2 / 1e3,
stat_cwt_bp['ocean'].slope * 60 *(co2.iloc[-1] - co2.iloc[0])**2 / 1e3)
