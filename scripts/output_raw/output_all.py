""" Outputs dataframes of spatial, yearly, decadal and whole time integrations (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell. """

import sys
sys.path.append("./../core/")

import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    if input_file.endswith(".pik"):
        input_file = pickle.load(open(input_file, 'rb'))
    
    df = inv_flux.TheDataFrame(data=input_file)
    
    df_spatial = df.spatial_integration()
    df_year = df_spatial.resample({'time': 'Y'}).sum()
    df_decade = df_spatial.resample({'time': '10Y'}).sum()
    df_whole = df_spatial.sum()
    
    
    try:
        os.mkdir(output_folder)
    except OSError:
        if os.path.isdir(output_folder):
            print("Directory %s already exists" % output_folder)
    else:
        print ("Successfully created the directory %s " % output_folder)
    
    
    # Output files after directory successfully created.
    pickle.dump(df_spatial, open("{}/spatial.pik".format(output_folder), "wb"))
    print ("Successfully created %s/spatial.pik " % output_folder)
    
    pickle.dump(df_year, open("{}/year.pik".format(output_folder), "wb"))
    print ("Successfully created %s/year.pik " % output_folder)
    
    pickle.dump(df_decade, open("{}/decade.pik".format(output_folder), "wb"))
    print ("Successfully created %s/decade.pik" % output_folder)
    
    pickle.dump(df_whole, open("{}/whole_time.pik".format(output_folder), "wb"))
    print ("Successfully created %s/whole_time.pik" % output_folder)
