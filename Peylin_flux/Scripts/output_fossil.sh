## USAGE bash output_fossil.sh
# For obtaining fossil_removed arrays of land and ocean fluxes for all 5 affected models.

CAMS="./../raw_data/fco2_CAMS-V17-1-2018_June2018-ext3_1979-2017_monthlymean_XYT.nc"
CTRACKER="./../raw_data/fco2_CTRACKER-EU-v2018_June2018-ext3_2000-2017_monthlymean_XYT.nc"
JAMSTEC="./../raw_data/fco2_JAMSTEC-V1-2-2018_June2018-ext3_1996-2017_monthlymean_XYT.nc"
JENA_s76="./../raw_data/fco2_JENA-s76-4-2-2018_June2018-ext3_1976-2017_monthlymean_XYT.nc"
JENA_s85="./../raw_data/fco2_JENA-s85-4-2-2018_June2018-ext3_1985-2017_monthlymean_XYT.nc"

python fossil_fix_data.py $CAMS ./../data_fixed/CAMS.pickle
python fossil_fix_data.py $CTRACKER ./../data_fixed/CTRACKER.pickle
python fossil_fix_data.py $JAMSTEC ./../data_fixed/JAMSTEC.pickle
python fossil_fix_data.py $JENA_s76 ./../data_fixed/JENA_s76.pickle
python fossil_fix_data.py $JENA_s85 ./../data_fixed/JENA_s85.pickle
