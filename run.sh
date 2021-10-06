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

# Run script
.venv/bin/python categories.py