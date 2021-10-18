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
from utils.logging.helpers import log_and_warn, log_initialize

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

    logging.info("Considering only claims located in Texas")
    texas_mask = (claim_data["state"] == "TX")

    if texas_mask.sum() > 0:
        log_and_warn("Total {}/{} claims are located in Texas".format(texas_mask.sum(),
                                                                    claim_data.shape[0],
                                                                ))
        claim_data = claim_data[texas_mask]
    else:
        logging.error("No claims are located in Texas")
        raise Exception("Claim processing error due to exception: no claims are located in Texas")

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

    loader.calculate_categories_stats(claim_data, category_regex, config["data"]["stats_file"])

    loader.calculate_claim_reports(claim_data, config["data"]["claim_report"], type="yearly")
    loader.calculate_claim_reports(claim_data, config["data"]["claim_report"], type="bp")

    loader.calculate_zip_code_mapping(claims=claim_data, column="category", filename=config["data"]["category_zip_data"].format(extension="xlsx"))
    loader.calculate_zip_code_mapping(claims=claim_data, column="subcategory", filename=config["data"]["subcategory_zip_data"].format(extension="xlsx"))

    primary_cities = ["Austin", "Dallas", "Houston"]
    
    zip_mapping_filename_path = config["data"]["category_zip_data"].split("/")
    zip_mapping_filename = "/".join(zip_mapping_filename_path[:-1]) + "/" + "_".join("_".join(city.split(" ")) for city in primary_cities) + "_" + zip_mapping_filename_path[-1].format(extension="xlsx")
    loader.calculate_zip_code_mapping(claims=claim_data, column="category", filename=zip_mapping_filename, primary_cities=primary_cities)

    zip_mapping_filename_path = config["data"]["subcategory_zip_data"].split("/")
    zip_mapping_filename = "/".join(zip_mapping_filename_path[:-1]) + "/" + "_".join("_".join(city.split(" ")) for city in primary_cities) + "_" + zip_mapping_filename_path[-1].format(extension="xlsx")
    loader.calculate_zip_code_mapping(claims=claim_data, column="subcategory", filename=zip_mapping_filename, primary_cities=primary_cities)

if __name__ == '__main__':
    main()