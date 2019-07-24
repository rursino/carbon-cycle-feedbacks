""" Outputs dataframes of spatial, yearly, decadal and whole time integrations (for all globe and regions) for each model.
Output format is binary csv through the use of pickle.

Run this script from the bash shell. """

import sys
import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    
    if input_file.endswith(".pickle"):
        input_file = pickle.load(open(input_file, 'rb'))
    
    df = inv_flux.TheDataFrame(data=input_file)
    
    df_spatial = df.spatial_integration()
    df_year = df.year_integration()
    df_decade = df.decade_integration()
    df_whole = df.whole_time_integration()
    
    
    try:
        os.mkdir(output_folder)
    except OSError:
        if os.path.isdir(output_folder):
            print("Directory %s already exists" % output_folder)
    else:
        print ("Successfully created the directory %s " % output_folder)
    
    
    # Output files after directory successfully created.
    pickle.dump(df_spatial, open("{}/spatial.pickle".format(output_folder), "wb"))
    print ("Successfully created %s/spatial.pickle " % output_folder)
    
    pickle.dump(df_year, open("{}/year.pickle".format(output_folder), "wb"))
    print ("Successfully created %s/year.pickle " % output_folder)
    
    pickle.dump(df_decade, open("{}/decade.pickle".format(output_folder), "wb"))
    print ("Successfully created %s/decade.pickle " % output_folder)
    
    pickle.dump(df_whole, open("{}/whole_time.pickle".format(output_folder), "wb"))
    print ("Successfully created %s/whole_time.pickle " % output_folder)
