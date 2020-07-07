## USAGE: bash output_all.sh
# to output dataframes of month, year, decade and whole time integrations of
# TEMP gridded temperature data for all globe and regions and for all datasets.

# Short-hand variables for the data paths of the 6 datasets.

CRUTEM="./../../../data/temp/crudata/CRUTEM.4.6.0.0.anomalies.nc"
HadCRUT="./../../../data/temp/crudata/HadCRUT.4.6.0.0.median.nc"
HadSST="./../../../data/temp/crudata/HadSST.3.1.1.0.anomalies.nc"


python output_all.py $MODEL ./../../../output/TEMP/spatial/output_all/CRUTEM/
python output_all.py $MODEL ./../../../output/TEMP/spatial/output_all/HadCRUT/
python output_all.py $MODEL ./../../../output/TEMP/spatial/output_all/HadSST/
