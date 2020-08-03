# Variables
# include config.mk
SPATIAL_DIR=./scripts/all_output/spatial
ANALYSIS_DIR=./scripts/all_output/analysis

# Clean cache files.
.PHONY : clean
clean :
	rm -rf tests/__pycache__ tests/.pytest_cache

# Output
.PHONY : spatial
spatial :
	bash $(SPATIAL_DIR)/*/output_all.sh
	# bash $(SPATIAL_DIR)/TRENDY/all_files.py

.PHONY : analysis
analysis :
	bash $(ANALYSIS_DIR)/inversions/analysis.sh
