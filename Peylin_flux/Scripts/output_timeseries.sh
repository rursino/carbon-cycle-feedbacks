## USAGE: bash output.sh
# to produce dataframes of land and ocean fluxes (global and annual)
# and then time series plots for land and ocean for all 6 models.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../data/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc"
CAMS="./../data_fixed/CAMS.pickle"
CTRACKER="./../data_fixed/CTRACKER.pickle"
JAMSTEC="./../data_fixed/JAMSTEC.pickle"
JENA_s76="./../data_fixed/JENA-s76.pickle"
JENA_s85="./../data_fixed/JENA-s85.pickle"


# Produce dataframes as csv files and time series plots.

python output_timeseries.py $Rayner ./../Output/output_Rayner.csv ./../Output/land_Rayner.png ./../Output/ocean_Rayner.png
python output_timeseries.py $CAMS ./../Output/output_CAMS.csv ./../Output/land_CAMS.png ./../Output/ocean_CAMS.png
python output_timeseries.py $CTRACKER ./../Output/output_CTRACKER.csv ./../Output/land_CTRACKER.png ./../Output/ocean_CTRACKER.png
python output_timeseries.py $JAMSTEC ./../Output/output_JAMSTEC.csv ./../Output/land_JAMSTEC.png ./../Output/ocean_JAMSTEC.png
python output_timeseries.py $JENA_s76 ./../Output/output_JENA_s76.csv ./../Output/land_JENA_s76.png ./../Output/ocean_JENA_s76.png
python output_timeseries.py $JENA_s85 ./../Output/output_JENA_s85.csv ./../Output/land_JENA_s85.png ./../Output/ocean_JENA_s85.png
