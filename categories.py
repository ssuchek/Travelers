import os
import inflect
import json
import logging

import config as constants
from config import config

import numpy as np
import pandas as pd

from utils.loader.loader import ClaimDataLoader

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

    claim_data = loader.preprocess_claims(claims=loaded_data, filename=config["data"]["preprocessed_file"].format(extension='csv'))

    word_frequency_data = loader.most_frequent_words(claim_data, config["data"]["word_frequency"])
    loader.plot_most_frequent_words(word_frequency_data, config["figures"]["word_frequency"])

    schema_file = config["data"]["categories_schema"]
    logging.info("Categories schema: {schema}".format(schema=schema_file))

    if not os.path.exists(schema_file):
        raise Exception("Categories initialisation: No categories specified")
    with open(schema_file, "r") as f:
        category_map = json.load(f)

    engine = inflect.engine()

    category_regex = {}
    for category, subcategory_map in category_map.items(): 
        category_regex[category] = {}
        for subcategory, patterns in subcategory_map.items():
            patterns.extend([engine.plural(p) for p in patterns])
            category_regex[category][subcategory] = "|".join([format_and_regex(p) for p in patterns])

    claim_data = loader.calculate_categories(claims=claim_data, categories=category_regex, filename=config["data"]["categorised_data"].format(extension="csv"))

    loader.calculate_categories_stats(claim_data, category_regex, config["data"]["stats_file"].format(extension="json"))

    category_zipcode_map = loader.calculate_zip_code_mapping(claims=claim_data, column="category", filename=config["data"]["category_zip_data"].format(extension="csv"))
    subcategory_zipcode_map = loader.calculate_zip_code_mapping(claims=claim_data, column="subcategory", filename=config["data"]["subcategory_zip_data"].format(extension="csv"))

if __name__ == '__main__':
    main()