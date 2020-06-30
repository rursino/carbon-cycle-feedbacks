## USAGE: bash output_all.sh
# to output dataframes of spatial, year, decade and whole time integrations
# for all globe and regions and for all 6 models.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../data/inversions/fco2_Rayner-C13-2018_June2018-ext3_1992-2012_monthlymean_XYT.nc"
CAMS="./../../data/inversions/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
CTRACKER="./../../data/inversions/fco2_CTRACKER-EU-v2018_June2018-ext3_2000-2017_monthlymean_XYT.nc"
JAMSTEC="./../../data/inversions/fco2_JAMSTEC-V1-2-2018_June2018-ext3_1996-2017_monthlymean_XYT.nc"
JENA_s76="./../../data/inversions/fco2_JENA-s76-4-2-2018_June2018-ext3_1976-2017_monthlymean_XYT.nc"
JENA_s85="./../../data/inversions/fco2_JENA-s85-4-2-2018_June2018-ext3_1985-2017_monthlymean_XYT.nc"


python output_all.py $Rayner ./../../../output/inversions/raw/output_all/Rayner_all/
python output_all.py $CAMS ./../../../output/inversions/raw/output_all/CAMS_all/
python output_all.py $CTRACKER ./../../../output/inversions/raw/output_all/CTRACKER_all/
python output_all.py $JAMSTEC ./../../../output/inversions/raw/output_all/JAMSTEC_all/
python output_all.py $JENA_s76 ./../../../output/inversions/raw/output_all/JENA_s76_all/
python output_all.py $JENA_s85 ./../../../output/inversions/raw/output_all/JENA_s85_all/
