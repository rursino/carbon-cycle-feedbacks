## USAGE: bash output.sh
# to produce dataframes of land and ocean fluxes (global and annual)
# and then time series plots for land and ocean for all 6 models.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../data/inversions/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc"
CAMS="./../../data/inversions/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
CTRACKER="./../../data/inversions/fco2_CTRACKER-EU-v2018_June2018-ext3_2000-2017_monthlymean_XYT.nc"
JAMSTEC="./../../data/inversions/fco2_JAMSTEC-V1-2-2018_June2018-ext3_1996-2017_monthlymean_XYT.nc"
JENA_s76="./../../data/inversions/fco2_JENA-s76-4-2-2018_June2018-ext3_1976-2017_monthlymean_XYT.nc"
JENA_s85="./../../data/inversions/fco2_JENA-s85-4-2-2018_June2018-ext3_1985-2017_monthlymean_XYT.nc"


# Produce dataframes as csv files and time series plots.

python output_timeseries.py $Rayner ./../../output/inversions/raw/output_Rayner.csv
python output_timeseries.py $CAMS ./../../output/inversions/raw/output_CAMS.csv
python output_timeseries.py $CTRACKER ./../../output/inversions/raw/output_CTRACKER.csv
python output_timeseries.py $JAMSTEC ./../../output/inversions/raw/output_JAMSTEC.csv
python output_timeseries.py $JENA_s76 ./../../output/inversions/raw/output_JENA_s76.csv
python output_timeseries.py $JENA_s85 ./../../output/inversions/raw/output_JENA_s85.csv
