## USAGE: bash analysis.sh
# to output analysis plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner_year="./../../../../output/inversions/spatial/output_all/Rayner/year.nc"
CAMS_year="./../../../../output/inversions/spatial/output_all/CAMS/year.nc"
CTRACKER_year="./../../../../output/inversions/spatial/output_all/CTRACKER/year.nc"
JAMSTEC_year="./../../../../output/inversions/spatial/output_all/JAMSTEC/year.nc"
JENA_s76_year_year="./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
JENA_s85="./../../../../output/inversions/spatial/output_all/JENA_s85/year.nc"

Rayner_monthly="./../../../../output/inversions/spatial/output_all/Rayner/spatial.nc"
CAMS_monthly="./../../../../output/inversions/spatial/output_all/CAMS/spatial.nc"
CTRACKER_monthly="./../../../../output/inversions/spatial/output_all/CTRACKER/spatial.nc"
JAMSTEC_monthly="./../../../../output/inversions/spatial/output_all/JAMSTEC/spatial.nc"
JENA_s76_monthly="./../../../../output/inversions/spatial/output_all/JENA_s76/spatial.nc"
JENA_s85_monthly="./../../../../output/inversions/spatial/output_all/JENA_s85/spatial.nc"


python analysis.py $Rayner_year Rayner 10 10
python analysis.py $CAMS_year CAMS 10 10
python analysis.py $CTRACKER_year CTRACKER 10 10
python analysis.py $JAMSTEC_year JAMSTEC 10 10
python analysis.py $JENA_s76_year JENA_s76 10 10
python analysis.py $JENA_s85_year JENA_s85 10 10

python analysis.py $Rayner_monthly Rayner 10 10
python analysis.py $CAMS_monthly CAMS 10 10
python analysis.py $CTRACKER_monthly CTRACKER 10 10
python analysis.py $JAMSTEC_monthly JAMSTEC 10 10
python analysis.py $JENA_s76_monthly JENA_s76 10 10
python analysis.py $JENA_s85_monthly JENA_s85 10 10
