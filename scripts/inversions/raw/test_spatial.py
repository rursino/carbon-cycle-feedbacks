import sys
sys.path.append("./../../core/")

import os
import inv_flux
import pickle

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_folder = sys.argv[2]

    if input_file.endswith(".pik"):
        input_file = pickle.load(open(input_file, 'rb'))

    df = inv_flux.SpatialAgg(data=input_file)

    df_spatial = df.spatial_integration()

    try:
        os.mkdir(output_folder)
    except OSError:
        if os.path.isdir(output_folder):
            print("Directory %s already exists" % output_folder)
    else:
        print ("Successfully created the directory %s " % output_folder)


    # Output files after directory successfully created.
    pickle.dump(df_spatial, open(f"{output_folder}/spatial.pik", "wb"))
    print ("Successfully created %s/spatial.pik " % output_folder)
