## USAGE: bash output_all.sh
# to output dataframes of year, decade and whole time integrations of TRENDY
# fluxes for all globe and regions and for all 6 models.

# Short-hand variables for the data paths of the 6 datasets.


LPJ_S0_cVeg="./../../../data/TRENDY/LPJ-GUESS/S0/LPJ-GUESS_S0_cVeg.nc"
LPJ_S1_cVeg="./../../../data/TRENDY/LPJ-GUESS/S1/LPJ-GUESS_S1_cVeg.nc"

python output_all.py $MODEL1 ./../../../output/TRENDY/spatial/output_all/LPJ_S0_cVeg/
python output_all.py $MODEL2 ./../../../output/TRENDY/spatial/output_all/LPJ_S1_cVeg/
