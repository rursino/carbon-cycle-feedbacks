## USAGE: bash model_evaluation.sh
# to output model evaluation plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../../output/inversions/raw/output_all/Rayner_all/year.pik"
CAMS="./../../../output/inversions/raw/output_all/CAMS_all/year.pik"
CTRACKER="./../../../output/inversions/raw/output_all/CTRACKER_all/year.pik"
JAMSTEC="./../../../output/inversions/raw/output_all/JAMSTEC_all/year.pik"
JENA_s76="./../../../output/inversions/raw/output_all/JENA_s76_all/year.pik"
JENA_s85="./../../../output/inversions/raw/output_all/JENA_s85_all/year.pik"


python model_evaluation.py $Rayner ./../../../output/inversions/model_evaluation/Rayner/
python model_evaluation.py $CAMS ./../../../output/inversions/model_evaluation/CAMS/
python model_evaluation.py $CTRACKER ./../../../output/inversions/model_evaluation/CTRACKER/
python model_evaluation.py $JAMSTEC ./../../../output/inversions/model_evaluation/JAMSTEC/
python model_evaluation.py $JENA_s76 ./../../../output/inversions/model_evaluation/JENA_s76/
python model_evaluation.py $JENA_s85 ./../../../output/inversions/model_evaluation/JENA_s85/
