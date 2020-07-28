""" Outputs dataframes of spatial, yearly, decadal and whole time integrations
(for all globe and regions) for each model. Output format is binary csv
through the use of pickle.

Run this script from the bash shell.
"""


""" IMPORTS """
import sys
sys.path.append("./../../../core/")
import inv_flux

import os
import xarray as xr
import pickle
import logging


""" FUNCTIONS """
def main(input_file, output_folder):
    """ Main function: to be used when script is not run from bash shell.
    """

    # Set up for logger to log success of results.
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename = 'result.log', level = logging.INFO,
                    format='%(asctime)s: %(levelname)s:%(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M')

    # Open dataset and run latitudinal_splits function.
    ds = xr.open_dataset(input_file)
    df = (inv_flux
            .SpatialAgg(data = ds)
            .latitudinal_splits()
         )

    arrays = {
        "month": df,
        "year": df.resample({'time': 'Y'}).sum(),
        "decade": df.resample({'time': '10Y'}).sum(),
        "whole": df.sum()
    }

    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    # Output files after directory successfully created.
    try:
        for freq in arrays:
            destination = f"{output_folder}/{freq}.nc"
            arrays[freq].to_netcdf(destination)

    except Exception as e:
        logger.error( '{} :: fail'.format(input_file.split('/')[-1]))
        logger.error(e)

    else:
        logger.info( '{} :: pass'.format(input_file.split('/')[-1]))


""" EXECUTION """
if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    main(input_file, output_folder)
