# Authors:     BP
# Maintainers: BP
# Copyright:   2022, HRDAG, GPL v2 or later
# =========================================
# sample_project/import/preproc/src/import.py

# dependencies
from pathlib import Path
from sys import stdout
import argparse
import logging
import numpy as np
import pandas as pd

# support methods
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compas", default=None)
    parser.add_argument("--cox_all", default=None)
    parser.add_argument("--cox_filt", default=None)
    parser.add_argument("--fairml", default=None)
    args = parser.parse_args()
    assert Path(args.compas).exists()
    assert Path(args.cox_all).exists()
    assert Path(args.cox_filt).exists()
    assert Path(args.fairml).exists()
    return args


def get_logger(sname, file_name=None):
    logger = logging.getLogger(sname)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s " +
                                  "- %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
    stream_handler = logging.StreamHandler(stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if file_name:
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def check_asserts(compas_df, cox_all_df, cox_filt_df, fairml_df):
    assert compas_df.shape == (60843, 28)
    assert cox_all_df.shape == (18316, 52)
    assert cox_filt_df.shape == (18316, 40)
    assert fairml_df.shape == (6172, 12)


# main
if __name__ == '__main__':

    # setup logging
    logger = get_logger(__name__, "output/import.log")

    # arg handling
    args = get_args()
    compas_f = args.compas
    cox_all_f = args.cox_all
    cox_filt_f = args.cox_filt
    fairml_f = args.fairml

    # read data, initial verification
    logging.info("Loading data.")
    compas_df = pd.read_csv(compas_f)
    cox_all_df = pd.read_csv(cox_all_f)
    cox_filt_df = pd.read_csv(cox_filt_f)
    fairml_df = pd.read_csv(fairml_f)
    check_asserts(compas_df, cox_all_df, cox_filt_df, fairml_df)
    
    # do stuff, more verification
    print('__main__ Summary:')
    print('compas_df shape:\t', compas_df.shape)
    print('cox_all shape:\t\t', cox_all_df.shape)
    print('cox_filt shape:\t\t', cox_filt_df.shape)
    print('fairml shape:\t\t', fairml_df.shape)
    print()
    compas_df.info()
    print()
    cox_all_df.info()
    print()
    cox_filt_df.info()
    print()
    fairml_df.info()

    # save data, final verification
    compas_df.to_parquet('output/compas.parquet')
    cox_all_df.to_parquet('output/cox_all.parquet')
    cox_filt_df.to_parquet('output/cox_filt.parquet')
    fairml_df.to_parquet('output/fairml.parquet')
    
    logging.info("done.")
    
# done.
    
# done.
