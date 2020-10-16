""" IMPORTS """
import matplotlib.pyplot as plt
from scipy import stats, signal
import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style('darkgrid')

import sys
from copy import deepcopy
from core import GCP_flux as GCPf

from importlib import reload
reload(GCPf);

import pickle


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
    plt.figure(figsize=(15,8))
    plt.plot(land, color='g')
    plt.plot(ocean, color='b')

    plt.title("GCP Uptake: 1959 - 2018", fontsize=32)
    plt.xlabel("Year", fontsize=24)
    plt.ylabel("Uptake flux to the atmosphere (GtC/yr)", fontsize=16)

    plt.legend(["land", "ocean"], fontsize=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_landocean.png")

    # For P.Rayner
    return pd.DataFrame({'land': land.values, 'ocean': ocean.values},
                        index=land.index
                        )

# RE-START HERE
def gcp_cwt(save=False, stat_values=False):
    zip_list = zip(['211', '212'], ['land', 'ocean'], [land_GCPf, ocean_GCPf])

    stat_vals = {}
    cwt_vals = {}
    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        sink_cwt_df = sink.cascading_window_trend(window_size=10)
        x = sink_cwt_df.index
        y = sink_cwt_df.values.squeeze()

        cwt_vals[var] = (pd
                            .DataFrame({'Year': x, 'CWT (GtC/yr/ppm)': y})
                            .set_index('Year')
                        )

        regstats = stats.linregress(x, y)
        slope, intercept, rvalue, pvalue, _ = regstats
        ax.plot(x, y)
        # ax.plot(x, x*slope + intercept)
        text = f"Slope: {(slope*1e3):.3f} MtC yr$^{'{-2}'}$ ppm$^{'{-1}'}$\nr = {rvalue:.3f}"
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
    else:
        # For P.Rayner
        return cwt_vals

def gcp_powerspec(save=False):
    zip_list = zip(['211', '212'], ['land', 'ocean'], [land_GCPf, ocean_GCPf])

    powerspec_df = {}

    fig = plt.figure(figsize=(16,8))
    axl = fig.add_subplot(111, frame_on=False)
    axl.tick_params(labelcolor="none", bottom=False, left=False)
    psd_vals = {}

    for subplot, var, sink in zip_list:
        ax = fig.add_subplot(subplot)

        sink_psd = sink.psd()
        x = sink_psd.iloc[:,0]
        y = sink_psd.iloc[:,1]

        psd_vals[var] = (pd
                            .DataFrame({'Period (years)': x.values,
                                        'Spectral Variance': y.values})
                            .set_index('Period (years)')
                        )

        ax.semilogy(x, y)
        ax.invert_xaxis()

    axl.set_title("Power Spectrum: GCP Uptake", fontsize=32, pad=20)
    axl.set_xlabel(sink_psd.columns[0], fontsize=16, labelpad=10)
    axl.set_ylabel(sink_psd.columns[1], fontsize=16,
                    labelpad=20)

    if save:
        plt.savefig(FIGURE_DIRECTORY + "gcp_powerspec.png")

    return psd_vals

def uptake_enso(save=False):
    enso = pd.read_csv("./../../data/climate_indices/soi_bom.csv", index_col="Year")
    enso.index = pd.date_range('1959-1', '2020-1', freq='M')
    soi_annual = enso.loc['1959-1':'2018-12'].resample('Y').mean().SOI

    soi_uptake = pd.DataFrame(
                    {'land': -land.values,
                     'ocean': -ocean.values,
                     'soi': soi_annual.values},
                    index=land.index)

    w = np.array([1/7, 1/2.01]) / (1 / 2) # Normalize the frequency.
    b, a = signal.butter(5, w, 'band')

    Bsoi = signal.filtfilt(b, a, soi_uptake['soi'])
    Bland = signal.filtfilt(b, a, soi_uptake['land'])
    Bocean = signal.filtfilt(b, a, soi_uptake['ocean'])
    # Bsoi = soi_uptake['soi']
    # Bland = soi_uptake['land']
    # Bocean = soi_uptake['ocean']

    fig = plt.figure(figsize=(14,8))
    ax1 = fig.add_subplot(211)
    ax1.plot(soi_uptake.index, Bland)
    ax2 = ax1.twinx()
    ax2.plot(soi_uptake.index, Bsoi, color='red')
    ax3 = fig.add_subplot(212)
    ax3.plot(soi_uptake.index, Bocean)
    ax4 = ax3.twinx()
    ax4.plot(soi_uptake.index, Bsoi, color='red')

    if save:
        plt.savefig(FIGURE_DIRECTORY + "uptake_enso_gcp.png")


""" EXECUTION """
gcp_landocean(save=False)

gcp_cwt(save=False, stat_values=True)

gcp_powerspec(save=False)

uptake_enso()

 # PICKLE
pickle.dump(gcp_landocean(), open(FIGURE_DIRECTORY+'/rayner_df/gcp_landocean.pik', 'wb'))
pickle.dump(gcp_cwt(), open(FIGURE_DIRECTORY+'/rayner_df/gcp_cwt.pik', 'wb'))
pickle.dump(gcp_powerspec(), open(FIGURE_DIRECTORY+'/rayner_df/gcp_powerspec.pik', 'wb'))
