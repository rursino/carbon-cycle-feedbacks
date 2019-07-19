# Clean out cache files.
.PHONY : clean
clean :
	rm Peylin_flux/Notebooks/.ipynb_checkpoints/*
	rm -d Peylin_flux/Notebooks/.ipynb_checkpoints/
	rm Prelim_Data_Analysis/Notebooks/.ipynb_checkpoints/*
	rm -d Prelim_Data_Analysis/Notebooks/.ipynb_checkpoints/
	rm Peylin_flux/Scripts/__pycache__/*
	rm -d Peylin_flux/Scripts/__pycache__/
	rm Peylin_flux/Scripts/*.pyc
