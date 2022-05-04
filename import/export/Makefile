# Authors:     BP
# Maintainers: BP
# Copyright:   2022, HRDAG, GPL v2 or later
# =========================================
# sample_project/import/export/Makefile

compas_in := ../preproc/output/compas.parquet
cox_all_in := ../preproc/output/cox_all.parquet
cox_filt_in := ../preproc/output/cox_filt.parquet
fairml_in := ../preproc/output/fairml.parquet

.PHONY: all clean compas_out cox_all_out cox_filt_out fairml_out

all: compas_out cox_all_out cox_filt_out fairml_out

clean: 
	-rm -r output/*

compas_out: $(compas_in)
	-mkdir output
	cd output && ln -s ../$< .

cox_all_out: \
		$(cox_all_in)
	-mkdir output
	cd output && ln -s ../$< .

cox_filt_out: \
		$(cox_filt_in)
	-mkdir output
	cd output && ln -s ../$< .

fairml_out: \
		$(fairml_in)
	-mkdir output
	cd output && ln -s ../$< .

# done.