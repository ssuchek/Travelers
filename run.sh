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

VERSION="v0.6.18"
# MATCHING_FILES="data/output/revised/single_year/$VERSION/"
# if ls $MATCHING_FILES* &> /dev/null; then
#     rm $MATCHING_FILES*
# fi

# PREPR_WEIGHTS_DB="data/preprocessed_weight_db_$VERSION"
# if ls $PREPR_WEIGHTS_DB* &> /dev/null; then
#     rm $PREPR_WEIGHTS_DB*
# fi

# FREQ_PATH=data/output/revised/single_year/$VERSION/
# if ls $FREQ_PATH*frequency* &> /dev/null; then
#     rm $FREQ_PATH*frequency*
# fi

# Run script
.venv/bin/python categories.py
.venv/bin/python calculate_weights.py
.venv/bin/python claim_id_weight_comparison.py --method="only_dumpster_loads" --no_axle
.venv/bin/python claim_id_weight_comparison.py --method="pickup_trucks" --no_axle
.venv/bin/python claim_id_weight_comparison.py --method="all_trucks" --no_axle
.venv/bin/python claim_id_weight_comparison.py --method="all_trucks" --axle