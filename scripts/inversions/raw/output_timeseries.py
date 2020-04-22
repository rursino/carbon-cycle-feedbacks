""" Obtains timeseries data of land and ocean fluxes (global & regional, annual) for each model from inv_flux.py module and saves it as a csv file.
If prompted, a plot of each of the global land and ocean timeseries is created and saved as png files.

Run this script is run from the shell. """

import sys
sys.path.append("./../core/")

import inv_flux
import xarray as xr
import pandas as pd

# import matplotlib.pyplot as plt
# from matplotlib.ticker import MaxNLocator


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if input_file.endswith(".pickle"):
        input_file = pickle.load(open(input_file, 'rb'))
    
    ds = inv_flux.TheDataFrame(data=input_file)
    
    # For plotting land and ocean fluxes (global, yearly).
    ds_year = ds.spatial_integration().resample({'time': 'Y'}).sum()
    
    year_array = [year.year for year in ds_year.time.values]
    
    df = pd.DataFrame({'Earth_Land': ds_year.Earth_Land.values,
                       'Earth_Ocean': ds_year.Earth_Ocean.values,
                       'South_Land': ds_year.South_Land.values,
                       'Tropical_Land': ds_year.Tropical_Land.values,
                       'North_Land': ds_year.North_Land.values,
                       'South_Ocean': ds_year.South_Ocean.values,
                       'Tropical_Ocean': ds_year.Tropical_Ocean.values,
                       'North_Ocean': ds_year.North_Ocean.values,
                       'Year': year_array
                      }).set_index('Year')
    
    df.to_csv(output_file) # Save dataframe as a csv file.
    print("Successfully saved file {}".format(output_file))
    
#     if len(sys.argv) > 3:
#         ax = plt.figure().gca()
#         ax.plot(df.index, df.Earth_Land)
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         plt.title("Annual Global Land Flux")
#         plt.xlabel("Year")
#         plt.ylabel("C flux to the atmosphere (GtC/yr)")
#         plt.ylim([-5,2])
#         plt.savefig(sys.argv[3])
#         print("Successfully saved plot {}".format(sys.argv[3]))
#     if len(sys.argv) > 4:
#         plt.clf()
#         ax = plt.figure().gca()
#         ax.plot(df.index, df.Earth_Ocean)
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#         plt.title("Annual Global Ocean Flux")
#         plt.xlabel("Year")
#         plt.ylabel("C flux to the atmosphere (GtC/yr)")
#         plt.ylim([-4,2])
#         plt.savefig(sys.argv[4])
#         print("Successfully saved plot {}".format(sys.argv[4]))
