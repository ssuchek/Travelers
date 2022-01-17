from genericpath import exists
import os
import inflect
import json
import logging

import nltk

try:
    nltk.find('corpora/wordnet')
except:
    nltk.download('wordnet')

try:
    nltk.find('corpora/stopwords')
except:
    nltk.download('stopwords')

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
    days_keep=7
)

def main():

    #raw_data_dir = config["data"]["raw_dir"]
    #claim_files = [os.path.join(raw_data_dir, f) for f in os.listdir(raw_data_dir) if os.path.isfile(os.path.join(raw_data_dir, f)) and "Xact_Categories" in f and f.endswith(".xlsx")]
    raw_data_dir = 'C:/Users/Florian/projects/Travelers/data/raw/single_year'
    claim_files = 'C:/Users/Florian/projects/Travelers/data/raw/single_year/Pentatonic_Xact_Categories_1yr_Final_1.xlsx', 'C:/Users/Florian/projects/Travelers/data/raw/single_year/Pentatonic_Xact_Categories_1yr_Final_2.xlsx'

    loader = ClaimDataLoader()

    logging.info("Loading data from files {}".format(claim_files))
    raw_data = loader.load_claim_data(filename=config["data"]["raw_file"].format(extension="csv"), input_files=claim_files)

    claim_data = loader.preprocess_claims(claims=raw_data, filename=config["data"]["preprocessed_data"].format(extension='csv'))

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

    # Load categories from schema. Throw exception if schema file does not exist
    schema_file = config["data"]["categories_schema"]
    logging.info("Categories schema: {schema}".format(schema=schema_file))

    if not os.path.exists(schema_file):
        raise Exception("Categories initialisation: No categories specified")

    with open(schema_file, "r") as f:
        category_map = json.load(f)

    # Categorize claim data
    logging.info("Categorizing data for Texas...")
    claim_data = loader.calculate_categories(claims=claim_data, categories=category_map, filename=config["data"]["categorised_data"].format(extension="csv"))

    weights_db_file = config["data"]["weights_db"]

    if os.path.exists(weights_db_file):
        weights_db = loader.preprocess_weights_db(weights_db_file=weights_db_file, filename=config["data"]["weights_preprocessed_db"])
        claims_weights_matching = loader.match_weights_db(weights=weights_db, claims=claim_data, filename=config["data"]["claims_weights_matching"])
        # loader.calculate_matched_match_weights_db(matched_claims=claims_weights_matching, primary_desc=primary_desc, categories=list(category_map), filename=config["data"]["claims_weights_matching_stats"])
    else:
        log_and_warn("No weights DB found!")
        weights_db = None

    # Analyze claims data based on different regions
    # primary_regions = ["Texas", "Austin", "Dallas", "Houston", "Other"]

    # for region in primary_regions:
    #     logging.info("Start processing data for {} region".format(region))

    #     os.makedirs(config["data"]["output_dir"].format(region=region), exist_ok=True)
    #     os.makedirs(config["figures"]["base_dir"].format(region=region), exist_ok=True)

    #     if region == "Texas":
    #         region_claim_data = claim_data.copy()

    #     elif region == "Other":
    #         region_mask = (claim_data["primary_city"].isin(primary_regions[:-1]))
    #         if region_mask.sum() > 0:
    #             log_and_warn("Total {}/{} claims are located in other regions".format((~region_mask).sum(),
    #                                                                         claim_data.shape[0]
    #                                                                     ))
    #             region_claim_data = claim_data[~region_mask]
    #         else:
    #             log_and_warn("No claims are located in other regions")
    #             continue
    #     else:
    #         region_mask = (claim_data["primary_city"] == region)
    #         if region_mask.sum() > 0:
    #             log_and_warn("Total {}/{} claims are located in {} region".format(region_mask.sum(),
    #                                                                         claim_data.shape[0],
    #                                                                         region
    #                                                                     ))
    #             region_claim_data = claim_data[region_mask]
    #         else:
    #             log_and_warn("No claims are located in {} region".format(region))
    #             continue

    #     # Word frequency analysis in claim description
    #     word_frequency_data = loader.most_frequent_words(region_claim_data, config["data"]["word_frequency"].format(region=region, extension="xlsx"))
    #     loader.plot_most_frequent_words(word_frequency_data, config["figures"]["word_frequency"].format(region=region))

    #     # Calculate the number of claims and items in each category and subcategory
    #     loader.calculate_categories_stats(region_claim_data, category_regex, config["data"]["stats_file"].format(region=region, extension="json"))

    #     # Calculate claim reports for each year and business/personal items
    #     loader.calculate_claim_reports(claims=region_claim_data, filename=config["data"]["claim_report"].format(region=region, extension="xlsx"), report_type="yearly")
    #     loader.calculate_claim_reports(claims=region_claim_data, filename=config["data"]["claim_report"].format(region=region, extension="xlsx"), report_type="bp")

    #     # Map ZIP codes to claim data
    #     loader.calculate_zip_code_mapping(claims=region_claim_data, column="category", filename=config["data"]["category_zip_data"].format(region=region, extension="xlsx"), primary_cities=[region])
    #     loader.calculate_zip_code_mapping(claims=region_claim_data, column="subcategory", filename=config["data"]["category_zip_data"].format(region=region, extension="xlsx"), primary_cities=[region])

    # primary_cities = ["Austin", "Dallas", "Houston"]
    
    # zip_mapping_filename_path = config["data"]["category_zip_data"].split("/")
    # zip_mapping_filename = "/".join(zip_mapping_filename_path[:-1]) + "/" + "_".join("_".join(city.split(" ")) for city in primary_cities) + "_" + zip_mapping_filename_path[-1].format(extension="xlsx")
    # loader.calculate_zip_code_mapping(claims=claim_data, column="category", filename=zip_mapping_filename, primary_cities=primary_cities)

    # zip_mapping_filename_path = config["data"]["subcategory_zip_data"].split("/")
    # zip_mapping_filename = "/".join(zip_mapping_filename_path[:-1]) + "/" + "_".join("_".join(city.split(" ")) for city in primary_cities) + "_" + zip_mapping_filename_path[-1].format(extension="xlsx")
    # loader.calculate_zip_code_mapping(claims=claim_data, column="subcategory", filename=zip_mapping_filename, primary_cities=primary_cities)

if __name__ == '__main__':
    main()