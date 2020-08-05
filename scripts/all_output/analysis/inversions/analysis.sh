## USAGE: bash analysis.sh
# to output analysis plots of all 6 model annual outputs.

# Short-hand variables for the data paths of the 6 datasets.

Rayner_year="./../../../../output/inversions/spatial/output_all/Rayner/year.nc"
CAMS_year="./../../../../output/inversions/spatial/output_all/CAMS/year.nc"
CTRACKER_year="./../../../../output/inversions/spatial/output_all/CTRACKER/year.nc"
JAMSTEC_year="./../../../../output/inversions/spatial/output_all/JAMSTEC/year.nc"
JENA_s76_year="./../../../../output/inversions/spatial/output_all/JENA_s76/year.nc"
JENA_s85_year="./../../../../output/inversions/spatial/output_all/JENA_s85/year.nc"

Rayner_month="./../../../../output/inversions/spatial/output_all/Rayner/month.nc"
CAMS_month="./../../../../output/inversions/spatial/output_all/CAMS/month.nc"
CTRACKER_month="./../../../../output/inversions/spatial/output_all/CTRACKER/month.nc"
JAMSTEC_month="./../../../../output/inversions/spatial/output_all/JAMSTEC/month.nc"
JENA_s76_month="./../../../../output/inversions/spatial/output_all/JENA_s76/month.nc"
JENA_s85_month="./../../../../output/inversions/spatial/output_all/JENA_s85/month.nc"

cd $(dirname $0)

python -W ignore analysis.py $Rayner_year 10 10
python -W ignore analysis.py $CAMS_year 10 10
python -W ignore analysis.py $CTRACKER_year 10 10
python -W ignore analysis.py $JAMSTEC_year 10 10
python -W ignore analysis.py $JENA_s76_year 10 10
python -W ignore analysis.py $JENA_s85_year 10 10

python -W ignore analysis.py $Rayner_month 10 10
python -W ignore analysis.py $CAMS_month 10 10
python -W ignore analysis.py $CTRACKER_month 10 10
python -W ignore analysis.py $JAMSTEC_month 10 10
python -W ignore analysis.py $JENA_s76_month 10 10
python -W ignore analysis.py $JENA_s85_month 10 10
