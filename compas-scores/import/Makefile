# Authors:     BP
# Maintainers: BP
# Copyright:   2022, HRDAG, GPL v2 or later
# =========================================
# sample_project/compas-scores/import/Makefile

HERE=$(shell git rev-parse --show-toplevel)

.PHONY: all clean compas_out

all: compas_out

clean: 
	-rm -r output/*

compas_out: $(HERE)/import/export/output/compas.parquet
	-mkdir output
	cd output && ln -s ../$< .

# done.