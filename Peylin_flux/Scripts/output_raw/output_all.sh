## USAGE: bash output_all.sh
# to output dataframes of spatial, year, decade and whole time integrations
# for all globe and regions and for all 6 models.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../raw_data/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc"
CAMS="./../../raw_data/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
CTRACKER="./../../raw_data/fco2_CTRACKER-EU-v2018_June2018-ext3_2000-2017_monthlymean_XYT.nc"
JAMSTEC="./../../raw_data/fco2_JAMSTEC-V1-2-2018_June2018-ext3_1996-2017_monthlymean_XYT.nc"
JENA_s76="./../../raw_data/fco2_JENA-s76-4-2-2018_June2018-ext3_1976-2017_monthlymean_XYT.nc"
JENA_s85="./../../raw_data/fco2_JENA-s85-4-2-2018_June2018-ext3_1985-2017_monthlymean_XYT.nc"


python output_all.py $Rayner ./../../Output/Rayner_all/
python output_all.py $CAMS ./../../Output/CAMS_all/
python output_all.py $CTRACKER ./../../Output/CTRACKER_all/
python output_all.py $JAMSTEC ./../../Output/JAMSTEC_all/
python output_all.py $JENA_s76 ./../../Output/JENA_s76_all/
python output_all.py $JENA_s85 ./../../Output/JENA_s85_all/
