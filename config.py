"""
Configuration parser and logging initializer
"""

import os
import logging
import configparser
import numpy as np
import pandas as pd
from pathlib import Path

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn


# load configuration file
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

# logging constants
LOG_FILE_MODE = config["logging"]["mode"]
LOG_LEVEL = getattr(logging, config["logging"]["level"].upper())
LOG_FORMAT = "%(asctime)s\t[%(levelname)s]\t%(message)s"

ADJECTIVES = {x.name().split('.', 1)[0] for x in wn.all_synsets('a')}
ADVERBS = {x.name().split('.', 1)[0] for x in wn.all_synsets('r')}
NOUNS = {x.name().split('.', 1)[0] for x in wn.all_synsets('n')}
VERBS = {x.name().split('.', 1)[0] for x in wn.all_synsets('v')}

STOP_WORDS = stopwords.words("english")
#  + ["per", "one", "two", "three", "four", "five", "six", 
#                                             "seven", "eight", "nine", "ten",
#                                             "lb", "kg", "lf", "sf", "yr",
#                                             "anti", "full", "large", "small", "high", "hour",
#                                             "period", "rfg",
#                                             "a", "the", "an", "s", "to"]

# STOP_CLAIMS = ["reset", "detach & reset", "detach and reset", "remove & reset", "remove and reset", "restore & repair", "restore and repair", 
#             "move", "clean", "cleaning", "content manipulation", "cont", "tear", "repair", "installed", "block and pad furniture", 
#             "furniture repair", "apperance allowance", "add", "fees", "labor", "fill in", "replace", "operator", "only labor",
#             "bulgary", "robbery", "mysterious disappearance", "civil unrest", "paint"]

STOP_CLAIMS = ["reset", "detach & reset", "detach and reset", "remove", "removal", "demolish/remove", "restore & repair", "restore and repair", 
            "move", "clean", "cleaning", "content manipulation", "cont", "repair", "installed", "block and pad furniture", 
            "furniture repair", "apperance allowance", "add", "labor", "fill in", "replace", "operator", "only labor",
            "bulgary", "robbery", "mysterious disappearance", "civil unrest", "paint", "painting", "permits", "remediation", 
            "treatment", "cont", "comp", "software", "tree", "dust control barrier per square foot", "floor protection heavy paper tape",
            "charge", "re-charge", "recharge", "installation", "dust control barrier", "debris chute", "floor prot", "floor protection",
            "clearance inspection", "landfil fees per tone", "debris disposal bid item", "mask cover"
            ]

STOP_CLAIMS_ADDITIONAL = ["calibration", "rewire", "evaluate", "provide", "return", "re-skin", "barrier", "repair", "installer", "clean-out", 
            "per week", "per day", "per month", "per invoice", "coating", "flashing"]



STOP_SMALL_ITEM_CLAIMS = ["fabric softener", "knick knacks"]
STOP_DONATED_REPURPOSED_CLAIMS = ["laptop cover", "camera bag", "artwork"]

PRIMARY_IGNORE_SUBCATEGORIES = [
    "cont", "fees", "excavation", "documents valuable papers", "firearms accessories", "health medical supplies",
    "interior lath plaster", "cash securities", "awnings patio covers", "setup", "personal care beauty", "perishable non-perishable"
]

SECONDARY_IGNORE_SUBCATEGORIES = [
    "finish hardware", "finish carpentry trimwork", "metal structure components", "misc equipment", "moisture protection",
    "steel components", "stucco", "wallpaper"
]

ALL_STOP_CLAIMS = STOP_CLAIMS + STOP_CLAIMS_ADDITIONAL + STOP_SMALL_ITEM_CLAIMS + STOP_DONATED_REPURPOSED_CLAIMS + PRIMARY_IGNORE_SUBCATEGORIES + SECONDARY_IGNORE_SUBCATEGORIES

FIELD_RENAME_MAP = {
    "Claim Identifier" : "claim_id",
    "category_description" : "subcategory_prev",
    "ITM_TYP_DESC_2" : "item_description",
    "DIV_CD" : "div_cd",
    "LS_YYMM" : "ls_date",
    "NOL_YYMM" : "nol_date",
    "peril_grp" : "reason",
    "ITM_UNIT_CD" : "item_unit_cd",
    "ITM_QTY" : "item_quantity",
    "count1" : "count"
}

ZIP_DB_FIELDS = [
    "zip", "primary_city", "state", "county", "housing_count", "population_count"
]

WEIGHTS_DB_FIELDS = [
    "pentatonic_id", 
    "waste_type", 
    "primary_desc", "secondary_desc", "material", "dimensions", "values_desc", 
    "unit", 
    "weight_lbs", "weight_ustons", "volume_cf", "volume_cy"
]

