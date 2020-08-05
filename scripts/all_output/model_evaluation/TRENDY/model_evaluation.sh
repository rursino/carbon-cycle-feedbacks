## USAGE: bash model_evaluation.sh
# to output model evaluation plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

CABLEPOP_S1="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S1_nbp/year.nc"
CLASSCTEM_S1="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S1_nbp/year.nc"
JSBACH_S1="./../../../../output/TRENDY/spatial/output_all/JSBACH_S1_nbp/year.nc"
OCN_S1="./../../../../output/TRENDY/spatial/output_all/OCN_S1_nbp/year.nc"
LPJGUESS_S1="./../../../../output/TRENDY/spatial/output_all/LPJ-GUESS_S1_nbp/year.nc"

CABLEPOP_S3="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S3_nbp/year.nc"
CLASSCTEM_S3="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S3_nbp/year.nc"
JSBACH_S3="./../../../../output/TRENDY/spatial/output_all/JSBACH_S3_nbp/year.nc"
OCN_S3="./../../../../output/TRENDY/spatial/output_all/OCN_S3_nbp/year.nc"
LPJGUESS_S3="./../../../../output/TRENDY/spatial/output_all/LPJ-GUESS_S3_nbp/year.nc"

WINDOW_SIZE=10

python model_evaluation.py $CABLEPOP_S1 $WINDOW_SIZE
python model_evaluation.py $CLASSCTEM_S1 $WINDOW_SIZE
python model_evaluation.py $JSBACH_S1 $WINDOW_SIZE
python model_evaluation.py $OCN_S1 $WINDOW_SIZE
python model_evaluation.py $LPJGUESS_S1 $WINDOW_SIZE

python model_evaluation.py $CABLEPOP_S3 $WINDOW_SIZE
python model_evaluation.py $CLASSCTEM_S3 $WINDOW_SIZE
python model_evaluation.py $JSBACH_S3 $WINDOW_SIZE
python model_evaluation.py $OCN_S3 $WINDOW_SIZE
python model_evaluation.py $LPJGUESS_S3 $WINDOW_SIZE
