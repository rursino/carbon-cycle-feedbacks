## USAGE: bash lin_reg_all.sh
# to output regression and plots of land and ocean uptake of all 6 models with decadal trends.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../Output/output_raw/output_all/Rayner_all/year.pik"
CAMS="./../Output/output_raw/output_all/CAMS_all/year.pik"
CTRACKER="./../Output/output_raw/output_all/CTRACKER_all/year.pik"
JAMSTEC="./../Output/output_raw/output_all/JAMSTEC_all/year.pik"
JENA_s76="./../Output/output_raw/output_all/JENA_s76_all/year.pik"
JENA_s85="./../Output/output_raw/output_all/JENA_s85_all/year.pik"


python lin_reg_all.py $Rayner Earth_Land ./lin_reg_times/Rayner_lrtime.pik ./../Output/output_linreg/Rayner/timeseries_land.png ./../Output/output_linreg/Rayner/regression_land.pik
python lin_reg_all.py $Rayner Earth_Ocean ./lin_reg_times/Rayner_lrtime.pik ./../Output/output_linreg/Rayner/timeseries_ocean.png ./../Output/output_linreg/Rayner/regression_ocean.pik


python lin_reg_all.py $CAMS Earth_Land ./lin_reg_times/CAMS_lrtime.pik ./../Output/output_linreg/CAMS/timeseries_land.png ./../Output/output_linreg/CAMS/regression_land.pik
python lin_reg_all.py $CAMS Earth_Ocean ./lin_reg_times/CAMS_lrtime.pik ./../Output/output_linreg/CAMS/timeseries_ocean.png ./../Output/output_linreg/CAMS/regression_ocean.pik


python lin_reg_all.py $CTRACKER Earth_Land ./lin_reg_times/CTRACKER_lrtime.pik ./../Output/output_linreg/CTRACKER/timeseries_land.png ./../Output/output_linreg/CTRACKER/regression_land.pik
python lin_reg_all.py $CTRACKER Earth_Ocean ./lin_reg_times/CTRACKER_lrtime.pik ./../Output/output_linreg/CTRACKER/timeseries_ocean.png ./../Output/output_linreg/CTRACKER/regression_ocean.pik


python lin_reg_all.py $JAMSTEC Earth_Land ./lin_reg_times/JAMSTEC_lrtime.pik ./../Output/output_linreg/JAMSTEC/timeseries_land.png ./../Output/output_linreg/JAMSTEC/regression_land.pik
python lin_reg_all.py $JAMSTEC Earth_Ocean ./lin_reg_times/JAMSTEC_lrtime.pik ./../Output/output_linreg/JAMSTEC/timeseries_ocean.png ./../Output/output_linreg/JAMSTEC/regression_ocean.pik


python lin_reg_all.py $JENA_s76 Earth_Land ./lin_reg_times/JENA_s76_lrtime.pik ./../Output/output_linreg/JENA_s76/timeseries_land.png ./../Output/output_linreg/JENA_s76/regression_land.pik
python lin_reg_all.py $JENA_s76 Earth_Ocean ./lin_reg_times/JENA_s76_lrtime.pik ./../Output/output_linreg/JENA_s76/timeseries_ocean.png ./../Output/output_linreg/JENA_s76/regression_ocean.pik


python lin_reg_all.py $JENA_s85 Earth_Land ./lin_reg_times/JENA_s85_lrtime.pik ./../Output/output_linreg/JENA_s85/timeseries_land.png ./../Output/output_linreg/JENA_s85/regression_land.pik
python lin_reg_all.py $JENA_s85 Earth_Ocean ./lin_reg_times/JENA_s85_lrtime.pik ./../Output/output_linreg/JENA_s85/timeseries_ocean.png ./../Output/output_linreg/JENA_s85/regression_ocean.pik