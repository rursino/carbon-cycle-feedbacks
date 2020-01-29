## USAGE: bash lin_reg_all.sh
# to output model evaluation plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../Output/output_raw/output_all/Rayner_all/year.pik"
CAMS="./../../Output/output_raw/output_all/CAMS_all/year.pik"
CTRACKER="./../../Output/output_raw/output_all/CTRACKER_all/year.pik"
JAMSTEC="./../../Output/output_raw/output_all/JAMSTEC_all/year.pik"
JENA_s76="./../../Output/output_raw/output_all/JENA_s76_all/year.pik"
JENA_s85="./../../Output/output_raw/output_all/JENA_s85_all/year.pik"


python model_evaluation.py $Rayner ./../../Output/model_evaluation/Rayner/
python model_evaluation.py $CAMS ./../../Output/model_evaluation/CAMS/
python model_evaluation.py $CTRACKER ./../../Output/model_evaluation/CTRACKER/
python model_evaluation.py $JAMSTEC ./../../Output/model_evaluation/JAMSTEC/
python model_evaluation.py $JENA_s76 ./../../Output/model_evaluation/JENA_s76/
python model_evaluation.py $JENA_s85 ./../../Output/model_evaluation/JENA_s85/
