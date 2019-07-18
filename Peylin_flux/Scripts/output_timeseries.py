""" Obtains timeseries data of land and ocean fluxes (global, annual) for all 6 models from inv_flux.py module and saves it as a csv file.
If prompted, a plot of each land and ocean timeseries is created and saved as png files.

Run this script is run from the shell. """

import sys
import inv_flux

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    df = inv_flux.TheDataFrame(data=input_file)
    
    # For plotting land and ocean fluxes (global, yearly).
    df_year = df.year_integration()
    output_df = df_year[['Year', 'earth_land_total', 'earth_ocean_total']]
    output_df.to_csv(output_file, index=False) # Save dataframe as a csv file.
    
    if len(sys.argv) > 3:
        ax = plt.figure().gca()
        ax.plot(output_df.Year, output_df.earth_land_total)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Annual Global Land Flux")
        plt.xlabel("Year")
        plt.ylabel("C flux to the atmosphere (GtC/yr)")
        plt.ylim([-5,11])
        plt.savefig(sys.argv[3])
    if len(sys.argv) > 4:
        plt.clf()
        ax = plt.figure().gca()
        ax.plot(output_df.Year, output_df.earth_ocean_total)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title("Annual Global Ocean Flux")
        plt.xlabel("Year")
        plt.ylabel("C flux to the atmosphere (GtC/yr)")
        plt.ylim([-5,2])
        plt.savefig(sys.argv[4])

