## USAGE: bash model_evaluation.sh
# to output model evaluation plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../../../output/inversions/spatial/output_all/Rayner/year.nc"
CAMS="./../../../../output/inversions/spatial/output_all/CAMS/year.nc"
CTRACKER="./../../../../output/inversions/spatial/output_all/CTRACKER/year.nc"
JAMSTEC="./../../../../output/inversions/spatial/output_all/JAMSTEC/year.nc"
JENA_s76="./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
JENA_s85="./../../../../output/inversions/spatial/output_all/JENA_s85/year.nc"


python model_evaluation.py $Rayner ./../../../../output/inversions/model_evaluation/Rayner/
python model_evaluation.py $CAMS ./../../../../output/inversions/model_evaluation/CAMS/
python model_evaluation.py $CTRACKER ./../../../../output/inversions/model_evaluation/CTRACKER/
python model_evaluation.py $JAMSTEC ./../../../../output/inversions/model_evaluation/JAMSTEC/
python model_evaluation.py $JENA_s76 ./../../../../output/inversions/model_evaluation/JENA_s76/
python model_evaluation.py $JENA_s85 ./../../../../output/inversions/model_evaluation/JENA_s85/
