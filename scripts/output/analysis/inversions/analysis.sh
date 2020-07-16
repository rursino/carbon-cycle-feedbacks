## USAGE: bash analysis.sh
# to output analysis plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner="./../../../../output/inversions/raw/output_all/Rayner_all/year.pik"
CAMS="./../../../../output/inversions/raw/output_all/CAMS_all/year.pik"
CTRACKER="./../../../../output/inversions/raw/output_all/CTRACKER_all/year.pik"
JAMSTEC="./../../../../output/inversions/raw/output_all/JAMSTEC_all/year.pik"
JENA_s76="./../../../../output/inversions/raw/output_all/JENA_s76_all/year.pik"
JENA_s85="./../../../../output/inversions/raw/output_all/JENA_s85_all/year.pik"

Rayner_monthly="./../../../../output/inversions/raw/output_all/Rayner_all/spatial.pik"
CAMS_monthly="./../../../../output/inversions/raw/output_all/CAMS_all/spatial.pik"
CTRACKER_monthly="./../../../../output/inversions/raw/output_all/CTRACKER_all/spatial.pik"
JAMSTEC_monthly="./../../../../output/inversions/raw/output_all/JAMSTEC_all/spatial.pik"
JENA_s76_monthly="./../../../../output/inversions/raw/output_all/JENA_s76_all/spatial.pik"
JENA_s85_monthly="./../../../../output/inversions/raw/output_all/JENA_s85_all/spatial.pik"


python analysis.py $Rayner Rayner 10 10
python analysis.py $CAMS CAMS 10 10
python analysis.py $CTRACKER CTRACKER 10 10
python analysis.py $JAMSTEC JAMSTEC 10 10
python analysis.py $JENA_s76 JENA_s76 10 10
python analysis.py $JENA_s85 JENA_s85 10 10

python analysis.py $Rayner_monthly Rayner 10 10
python analysis.py $CAMS_monthly CAMS 10 10
python analysis.py $CTRACKER_monthly CTRACKER 10 10
python analysis.py $JAMSTEC_monthly JAMSTEC 10 10
python analysis.py $JENA_s76_monthly JENA_s76 10 10
python analysis.py $JENA_s85_monthly JENA_s85 10 10
