import os
import re
import sys
import json
import xlrd
import logging

import config as constants
from config import config

import numpy as np
import pandas as pd

from utils.loader.loader import ClaimDataLoader

from utils.preprocess import Preprocessor, PreprocessTransformation
from utils.preprocess import BASIC_PREPROCESS

from utils.helpers import format_and_regex
from utils.logging.helpers import log_initialize

log_initialize(
    file_path=config["logging"]["log_path_claim_data"],
    file_mode=constants.LOG_FILE_MODE,
    log_level=constants.LOG_LEVEL,
    log_format_str=constants.LOG_FORMAT,
    days_keep=30
)

def main():

    raw_data_dir = config["data"]["raw_dir"]

    claim_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f)) and "Xact_Categories" in f and f.endswith(".xlsx")]

    loader = ClaimDataLoader()

    logging.info("Loading data from files {}".format(claim_files))
    loaded_data = loader.load_claim_data(filename=config["data"]["raw_file"].format(extension="csv"), input_files=claim_files)

    basic_preprocessor = Preprocessor(BASIC_PREPROCESS)
    claim_data = basic_preprocessor.calculate(loaded_data)

    word_frequency_data = loader.most_frequent_words(claim_data, config["data"]["word_frequency"])
    loader.plot_most_frequent_words(word_frequency_data, config["figures"]["word_frequency"])

    schema_file = config["data"]["categories_schema"]
    logging.info("Categories schema: {schema}".format(schema=schema_file))

    if not os.path.exists(schema_file):
        raise Exception("Categories initialisation: No categories specified")
    with open(schema_file, "r") as f:
        category_map = json.load(f)

    category_regex = {}
    for category, subcategory_map in category_map.items(): 
        category_regex[category] = {}
        for subcategory, patterns in subcategory_map.items():
            category_regex[category][subcategory] = "|".join([format_and_regex(p) for p in patterns])

    claim_data = loader.calculate_categories(categories=category_regex, filename=config["data"]["categorised_data"].format(extension="csv"))

    loader.calculate_categories_stats(claim_data, category_regex, config["data"]["stats_file"].format(extension="json"))

if __name__ == '__main__':
    main()