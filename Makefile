# Clean cache files.
.PHONY : clean
clean :
	rm -rf tests/__pycache__ tests/.pytest_cache

.PHONY : spatial
spatial :
	bash scripts/all_output/spatial/inversions/output_all.sh
	bash scripts/all_output/spatial/TEMP/output_all.sh
	python scripts/all_output/spatial/TRENDY/all_files.py
	python scripts/all_output/spatial/TRENDY/mean_TRENDY.py

.PHONY : analysis
analysis : spatial
	bash ./scripts/all_output/analysis/*/analysis.sh

.PHONY : model_evaluation
model_evaluation : spatial
	bash ./scripts/all_output/analysis/*/analysis.sh

feedbacks :
	cd ./scripts/all_output/feedbacks/
	python output.py
	python further_output.py
	cd ../../../
