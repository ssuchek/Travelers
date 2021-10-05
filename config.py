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

# load configuration file
config = configparser.ConfigParser()
config.read(os.path.join(os.path.dirname(__file__), "config.ini"))

# logging constants
LOG_FILE_MODE = config["logging"]["mode"]
LOG_LEVEL = getattr(logging, config["logging"]["level"].upper())
LOG_FORMAT = "%(asctime)s\t[%(levelname)s]\t%(message)s"

STOP_WORDS = stopwords.words('english') + ['per', 'one', 'two', 'three', 'four', 'five', 'six', 
                                                   'seven', 'eight', 'nine', 'ten',
                                                  'lb', 'kg', 'lf', 'sf', 'yr',
                                                  'anti', 'full', 'large', 'small', 'high', 'hour',
                                                  'period', 'rfg']

STOP_CLAIMS = ["reset", "detach & reset", "remove & reset", "restore & repair", "move", "clean", 
            "cleaning", "content manipulation", "cont", "tear", "repair", "installed", "block and pad furniture", 
            "furniture repair", "apperance allowance", "add", "fees", "labor", "fill in", "replace",
            "bulgary", "robbery", "mysterious disappearance", "civil unrest"]

STOP_SMALL_ITEM_CLAIMS = ["fabric softener", "knick knacks"]

ALL_STOP_WORDS = STOP_WORDS + STOP_CLAIMS + STOP_SMALL_ITEM_CLAIMS

FIELD_RENAME_MAP = {
    "category_description" : "subcategory_prev",
    "ITM_TYP_DESC_2" : "item_description",
    "DIV_CD" : "div_cd",
    "LS_YYMM" : "ls_date",
    "peril_grp" : "reason",
    "ITM_UNIT_CD" : "item_unit_cd",
    "ITM_QTY" : "item_quantity",
    "count1" : "count"
}

