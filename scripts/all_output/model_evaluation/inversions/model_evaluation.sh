## USAGE: bash model_evaluation.sh
# to output model evaluation plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../../../output/inversions/spatial/output_all/Rayner/year.nc"
CAMS="./../../../../output/inversions/spatial/output_all/CAMS/year.nc"
CTRACKER="./../../../../output/inversions/spatial/output_all/CTRACKER/year.nc"
JAMSTEC="./../../../../output/inversions/spatial/output_all/JAMSTEC/year.nc"
JENA_s76="./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
JENA_s85="./../../../../output/inversions/spatial/output_all/JENA_s85/year.nc"

WINDOW_SIZE=10

python model_evaluation.py $Rayner $WINDOW_SIZE
python model_evaluation.py $CAMS $WINDOW_SIZE
python model_evaluation.py $CTRACKER $WINDOW_SIZE
python model_evaluation.py $JAMSTEC $WINDOW_SIZE
python model_evaluation.py $JENA_s76 $WINDOW_SIZE
python model_evaluation.py $JENA_s85 $WINDOW_SIZE
