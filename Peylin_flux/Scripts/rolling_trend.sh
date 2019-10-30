## USAGE: bash rolling_trend.sh
# to output rolling trend and plots of land and ocean uptake of all 6 models.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../Output/output_raw/output_all/Rayner_all/year_wco2.pik"
CAMS="./../Output/output_raw/output_all/CAMS_all/year_wco2.pik"
CTRACKER="./../Output/output_raw/output_all/CTRACKER_all/year_wco2.pik"
JAMSTEC="./../Output/output_raw/output_all/JAMSTEC_all/year_wco2.pik"
JENA_s76="./../Output/output_raw/output_all/JENA_s76_all/year_wco2.pik"
JENA_s85="./../Output/output_raw/output_all/JENA_s85_all/year_wco2.pik"


python rolling_trend.py $Rayner Earth_Land ./../Output/output_linreg/Rayner/roll_plot_land.png ./../Output/output_linreg/Rayner/roll_df_land.pik
python rolling_trend.py $Rayner Earth_Ocean ./../Output/output_linreg/Rayner/roll_plot_ocean.png ./../Output/output_linreg/Rayner/roll_df_ocean.pik


python rolling_trend.py $CAMS Earth_Land ./../Output/output_linreg/CAMS/roll_plot_land.png ./../Output/output_linreg/CAMS/roll_df_land.pik
python rolling_trend.py $CAMS Earth_Ocean ./../Output/output_linreg/CAMS/roll_plot_ocean.png ./../Output/output_linreg/CAMS/roll_df_ocean.pik


python rolling_trend.py $CTRACKER Earth_Land ./../Output/output_linreg/CTRACKER/roll_plot_land.png ./../Output/output_linreg/CTRACKER/roll_df_land.pik
python rolling_trend.py $CTRACKER Earth_Ocean ./../Output/output_linreg/CTRACKER/roll_plot_ocean.png ./../Output/output_linreg/CTRACKER/roll_df_ocean.pik


python rolling_trend.py $JAMSTEC Earth_Land ./../Output/output_linreg/JAMSTEC/roll_plot_land.png ./../Output/output_linreg/JAMSTEC/roll_df_land.pik
python rolling_trend.py $JAMSTEC Earth_Ocean ./../Output/output_linreg/JAMSTEC/roll_plot_ocean.png ./../Output/output_linreg/JAMSTEC/roll_df_ocean.pik


python rolling_trend.py $JENA_s76 Earth_Land ./../Output/output_linreg/JENA_s76/roll_plot_land.png ./../Output/output_linreg/JENA_s76/roll_df_land.pik
python rolling_trend.py $JENA_s76 Earth_Ocean ./../Output/output_linreg/JENA_s76/roll_plot_ocean.png ./../Output/output_linreg/JENA_s76/roll_df_ocean.pik


python rolling_trend.py $JENA_s85 Earth_Land ./../Output/output_linreg/JENA_s85/roll_plot_land.png ./../Output/output_linreg/JENA_s85/roll_df_land.pik
python rolling_trend.py $JENA_s85 Earth_Ocean ./../Output/output_linreg/JENA_s85/roll_plot_ocean.png ./../Output/output_linreg/JENA_s85/roll_df_ocean.pik