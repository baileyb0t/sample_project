# Authors:     BP
# Maintainers: BP
# Copyright:   2022, HRDAG, GPL v2 or later
# =========================================
# sample_project/import/preproc/Makefile

compas_f = 'input/compas-scores-raw.csv'
cox_all_f = 'input/cox-violent-parsed.csv'
cox_filt_f = 'input/cox-violent-parsed_filt.csv'
fairml_f = 'input/propublica_data_for_fairml.csv'

.PHONY: all clean output

all: output

clean: 
	-rm -r output/*

output: \
		src/import.py \
		$(input)
	-mkdir output
	python3 $< \
		--compas=$(compas_f) \
		--cox_all=$(cox_all_f) \
		--cox_filt=$(cox_filt_f) \
		--fairml=$(fairml_f)

# done.