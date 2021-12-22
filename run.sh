#!/usr/bin/env bash

# Install Python packages 
.venv/bin/python -m pip install -r requirements.txt

# [Optional] Remove data file before to recreate them
# if ls data/output/* &> /dev/null; then
#     rm data/output/*
# fi

# if ls figs/* &> /dev/null; then
#     rm figs/*
# fi
MATCHING_FILES="data/output/revised/single_year/claims_weight_db_matched"
if ls $MATCHING_FILES* &> /dev/null; then
    rm $MATCHING_FILES*
fi

PREPR_WEIGHTS_DB="data/preprocessed_weight_db_v0.5.2.csv"
if [[ -f $PREPR_WEIGHTS_DB ]]; then
    rm $PREPR_WEIGHTS_DB
fi

# Run script
.venv/bin/python categories.py