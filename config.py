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

STOP_CLAIMS = [
                "add", "appearance allowance",
                "barrier", "block pad furniture"
                "calibration", "charge", "civil unrest", "clean", "cleaning", "clean-out", "cleanout", "clearance inspection", "coating", "comp", "cont", "content manipulation",
                "debris chute", "debris disposal bid item", "demolish/remove", "detach reset", "dust control barrier",
                "fill in", "flashing", "floor prot", "floor protection", "furniture repair", 
                "evaluate",
                "installation", "installed", "installer",
                "labor", "landfil fees",
                "mask cover", "move",
                "operator", 
                "paint", "painting", "permits", "per day", "per invoice", "per month", "per week", "provide",
                "re-charge", "recharge", "remediation", "remove", "removal", "repair", "replace", "reset", "re-skin", "reskin", "restore repair", "return", "rewire",
                "software",
                "treatment", "tree"
            ]

STOP_SMALL_ITEM_CLAIMS = ["fabric softener", "knick knacks"]
STOP_DONATED_REPURPOSED_CLAIMS = ["laptop cover", "camera bag", "artwork"]

STOP_CATEGORIES = [
    "fees", "excavation", "documents valuable papers", "firearms accessories", "health medical supplies",
    "interior lath plaster", "cash securities", "awnings patio covers", "setup", "personal care beauty", "perishable non-perishable",
    "finish hardware", "finish carpentry trimwork", "metal structure components", "misc equipment", "moisture protection",
    "steel components", "stucco", "wallpaper"
]

STOP_REASON_DESC = [
    "burglary", "mysterious disappearance", "robbery"
]

ALL_STOP_CLAIMS = STOP_CLAIMS + STOP_SMALL_ITEM_CLAIMS + STOP_DONATED_REPURPOSED_CLAIMS

KEEP_ITEMS = [
    "haul", "dispose"
]

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

