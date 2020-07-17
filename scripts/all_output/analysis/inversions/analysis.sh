## USAGE: bash analysis.sh
# to output analysis plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner_year="./../../../../output/inversions/spatial/output_all/Rayner/year.nc"
CAMS_year="./../../../../output/inversions/spatial/output_all/CAMS/year.nc"
CTRACKER_year="./../../../../output/inversions/spatial/output_all/CTRACKER/year.nc"
JAMSTEC_year="./../../../../output/inversions/spatial/output_all/JAMSTEC/year.nc"
JENA_s76_year_year="./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
JENA_s85="./../../../../output/inversions/spatial/output_all/JENA_s85/year.nc"

Rayner_month="./../../../../output/inversions/spatial/output_all/Rayner/month.nc"
CAMS_month="./../../../../output/inversions/spatial/output_all/CAMS/month.nc"
CTRACKER_month="./../../../../output/inversions/spatial/output_all/CTRACKER/month.nc"
JAMSTEC_month="./../../../../output/inversions/spatial/output_all/JAMSTEC/month.nc"
JENA_s76_month="./../../../../output/inversions/spatial/output_all/JENA_s76/month.nc"
JENA_s85_month="./../../../../output/inversions/spatial/output_all/JENA_s85/month.nc"


python analysis.py $Rayner_year Rayner 10 10
python analysis.py $CAMS_year CAMS 10 10
python analysis.py $CTRACKER_year CTRACKER 10 10
python analysis.py $JAMSTEC_year JAMSTEC 10 10
python analysis.py $JENA_s76_year JENA_s76 10 10
python analysis.py $JENA_s85_year JENA_s85 10 10

python analysis.py $Rayner_month Rayner 10 10
python analysis.py $CAMS_month CAMS 10 10
python analysis.py $CTRACKER_month CTRACKER 10 10
python analysis.py $JAMSTEC_month JAMSTEC 10 10
python analysis.py $JENA_s76_month JENA_s76 10 10
python analysis.py $JENA_s85_month JENA_s85 10 10
