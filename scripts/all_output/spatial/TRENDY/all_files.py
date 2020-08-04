""" Outputs file names to use as input for spatial/output_all.py and executes
this script for each input file.
"""

""" IMPORTS """
import os
from itertools import *
import output_all
import logging
from tqdm import tqdm


""" SETUP """
CURRENT_DIR = os.path.dirname(__file__)
output_dir = CURRENT_DIR + "./../../../../output/TRENDY/spatial/output_all/"

logger = logging.getLogger(__name__)
logging.basicConfig(filename = CURRENT_DIR + './result.log', level = logging.INFO,
                    format='%(asctime)s: %(levelname)s:%(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M')


""" FUNCTIONS """
def get_subnames(subname_type):
    """ Get sub-names (for .nc filenames) from any one of the following types
    (pass in subname_type parameter):

        variables
        models
        simulations

    """

    for index, line in enumerate(infoContent):
        if subname_type.upper() in line:
            subname_index = index
            break

    subnames = []
    subname_index += 2
    while True:
        subname_line = infoContent[subname_index]
        if '-' * 24 in subname_line:
            break
        elif subname_line and (not subname_line.startswith('!')):
            subnames.append(subname_line.split()[0])
        subname_index += 1

    return subnames

def generate_filename(name):
    """ Generate a filename from a set of inputs, passed as 'name'.
    """

    model, simulation, variable = name
    return f"{model}/{simulation}/{model}_{simulation}_{variable}.nc"


""" EXECUTION """
if __name__ == "__main__":

    info_fname = CURRENT_DIR + './../../../../data/TRENDY/models/info.txt'
    with open(info_fname, 'r') as info:
        infoContent = info.read().splitlines()

    variables = get_subnames('variables')
    models = get_subnames('models')
    simulations = get_subnames('simulations')

    product_subnames = product(models, simulations, variables)

    for name in tqdm(product_subnames):
        input_file = (CURRENT_DIR + './../../../../data/TRENDY/models/' +
                      generate_filename(name))
        output_folder = output_dir + "{}_{}_{}/".format(*name)
        try:
            output_all.main(input_file, output_folder)
        except Exception as e:
            logger.error(' {}_{}_{} :: fail'.format(*name))
            logger.error(e)
        else:
            logger.info(' {}_{}_{}:: pass'.format(*name))
