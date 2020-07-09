""" Outputs file names to use as input for spatial/output_all.py and executes
this script for each input file.
"""

""" IMPORTS """
from itertools import *
import output_all

""" SETUP """
output_dir = "./../../../output/TRENDY/spatial/output_all/"


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
    model, simulation, variable = name
    return f"{model}/{simulation}/{model}_{simulation}_{variable}.nc"


""" EXECUTION """
if __name__ == "__main__":

    with open('./../../../data/TRENDY/models/info.txt', 'r') as info:
        infoContent = info.read().splitlines()

    variables = get_subnames('variables')
    models = get_subnames('models')
    simulations = get_subnames('simulations')

    product_subnames = product(models, simulations, variables)

    for name in product_subnames:
        input_file = './../../../data/TRENDY/models/' + generate_filename(name)
        output_folder = output_dir + "{}_{}_{}/".format(*name)
        output_all.main(input_file, output_folder)
