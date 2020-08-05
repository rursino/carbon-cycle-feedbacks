## USAGE: bash analysis.sh
# to output analysis plots of all 5 TRENDY model outputs.

# Short-hand variables for the data paths of the 5 datasets.

CABLEPOP_S1_year="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S1_nbp/year.nc"
CLASSCTEM_S1_year="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S1_nbp/year.nc"
JSBACH_S1_year="./../../../../output/TRENDY/spatial/output_all/JSBACH_S1_nbp/year.nc"
OCN_S1_year="./../../../../output/TRENDY/spatial/output_all/OCN_S1_nbp/year.nc"
LPJGUESS_S1_year="./../../../../output/TRENDY/spatial/output_all/LPJ-GUESS_S1_nbp/year.nc"

CABLEPOP_S1_month="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S1_nbp/month.nc"
CLASSCTEM_S1_month="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S1_nbp/month.nc"
JSBACH_S1_month="./../../../../output/TRENDY/spatial/output_all/JSBACH_S1_nbp/month.nc"
OCN_S1_month="./../../../../output/TRENDY/spatial/output_all/OCN_S1_nbp/month.nc"

CABLEPOP_S3_year="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S3_nbp/year.nc"
CLASSCTEM_S3_year="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S3_nbp/year.nc"
JSBACH_S3_year="./../../../../output/TRENDY/spatial/output_all/JSBACH_S3_nbp/year.nc"
OCN_S3_year="./../../../../output/TRENDY/spatial/output_all/OCN_S3_nbp/year.nc"
LPJGUESS_S3_year="./../../../../output/TRENDY/spatial/output_all/LPJ-GUESS_S3_nbp/year.nc"

CABLEPOP_S3_month="./../../../../output/TRENDY/spatial/output_all/CABLE-POP_S3_nbp/month.nc"
CLASSCTEM_S3_month="./../../../../output/TRENDY/spatial/output_all/CLASS-CTEM_S3_nbp/month.nc"
JSBACH_S3_month="./../../../../output/TRENDY/spatial/output_all/JSBACH_S3_nbp/month.nc"
OCN_S3_month="./../../../../output/TRENDY/spatial/output_all/OCN_S3_nbp/month.nc"

mean_S1_year="./../../../../output/TRENDY/spatial/mean_all/S1/year.nc"
mean_S1_month="./../../../../output/TRENDY/spatial/mean_all/S1/month.nc"
mean_S3_year="./../../../../output/TRENDY/spatial/mean_all/S3/year.nc"
mean_S3_month="./../../../../output/TRENDY/spatial/mean_all/S3/month.nc"

WINDOW_SIZE=10
PERIOD=10

cd $(dirname $0)

# python -W ignore analysis.py $CABLEPOP_S1_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $CLASSCTEM_S1_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $JSBACH_S1_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $OCN_S1_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $LPJGUESS_S1_year $WINDOW_SIZE $PERIOD
#
# python -W ignore analysis.py $CABLEPOP_S1_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $CLASSCTEM_S1_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $JSBACH_S1_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $OCN_S1_month $WINDOW_SIZE $PERIOD
#
# python -W ignore analysis.py $CABLEPOP_S3_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $CLASSCTEM_S3_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $JSBACH_S3_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $OCN_S3_year $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $LPJGUESS_S3_year $WINDOW_SIZE $PERIOD
#
# python -W ignore analysis.py $CABLEPOP_S3_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $CLASSCTEM_S3_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $JSBACH_S3_month $WINDOW_SIZE $PERIOD
# python -W ignore analysis.py $OCN_S3_month $WINDOW_SIZE $PERIOD

python -W ignore analysis.py $mean_S1_year $WINDOW_SIZE $PERIOD
python -W ignore analysis.py $mean_S1_month $WINDOW_SIZE $PERIOD
python -W ignore analysis.py $mean_S3_year $WINDOW_SIZE $PERIOD
python -W ignore analysis.py $mean_S3_month $WINDOW_SIZE $PERIOD
