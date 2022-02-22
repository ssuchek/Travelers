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

FREQUENCY_STOP_WORDS = STOP_WORDS + [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "lb", "kg", "lf", "sf", "yr", "sq", "tons", "yards",
    "large", "medium", "small", "high", "low", "tall",
    "anti", "full", "hour", "period", "rfg", "per", "approx"
]

STOP_CLAIMS = [
                "& finish",
                "add", "adhesive", "and finish", "applied", "appearance allowance", "assembly",
                "barrier", "bid item", "block pad furniture", "buff",
                "calibration", "charge", "civil unrest", "clean", "cleaning", "clean-out", "cleanout", 
                "clearance inspection", "coating", "comp", "cont", "content manipulation",
                "debris chute", "demolish remove", "detach reset", "dust control barrier",
                "finish", "fill", "flashing", "floor prot", "floor protection", "furniture repair", 
                "equipment wiring", "evaluate",
                "hot mopped",
                "installation", "installed", "installer",
                "labor", "landfil fees", "loading",
                "mask", "modified", "molding", "move", "mulching",
                "open and close", "operator", 
                "paint", "painting", "permits", "per day", "per invoice", "per month", "per week", "plumbing fixture", "polish", 
                "preparation", "primed", "primer", "prep", "protect", "provide",
                "recharge", "refacing", "reglaze", "regrout", "reinforced", "remediation", "remove", "removal", 
                "repair", "replace", "reset", "rescreen", "reskin", "restore repair", "return", "rewire", "roll roofing",
                "sand", "scarify", "seal", "sealer", "simulated", "software", "spray",
                "tape joint", "tape joint repair", "tear", "test", "texture", "tongue groove", "treatment", "tree",
                "wash"
            ]

STOP_CLAIMS_CONTAINING = ["re-"]

STOP_CLAIMS_PRECISE_MATCHING = ["plumbing"]

STOP_SMALL_ITEM_CLAIMS = ["fabric softener", "knick knacks"]
STOP_DONATED_REPURPOSED_CLAIMS = ["laptop cover", "camera bag", "artwork"]

STOP_CATEGORIES = [
    "awnings patio covers", 
    "cash securities",
    "documents valuable papers",
    "excavation",  
    "fees", "finish carpentry trimwork", "finish hardware", "firearms accessories", 
    "health medical supplies",
    "interior lath plaster", 
    "metal structure components", "misc equipment", "moisture protection",
    "paneling wood wall finishes", "perishable non-perishable", "personal care beauty",   
    "setup", "steel components", "stucco",
    "wallpaper"
]

STOP_REASON_DESC = [
    "burglary", "mysterious disappearance", "robbery"
]

STOP_ACTIVITIES = [
    "ADDITIONAL COST FOR HIGH WALL OR CEILING - 11' TO 14' HIGH",
    "ADDITIONAL COST FOR HIGH WALL OR CEILING - OVER 14' TO 20'",
    'ASPHALT STARTER - PEEL AND STICK',
    '''BATT INSULATION REPLACEMENT PER LF - 4" - UP TO 2' TALL''',
    'CAULKING - SILICONE',
    'CONCRETE CUTTING - SLAB (PER LF PER INCH OF SAW DEPTH)',
    'CONCRETE FLOOR SAWING - 4" SLAB',
    'CONCRETE SEALER - BRUSH OR SPRAY APPLIED',
    'DUCTWORK - ADD-ON FOR CONFINED SPACES - FLEXIBLE',
    'EPOXY CRACK AND JOINT FILLER (PER LF OF CRACK)',
    'EXTERIOR LIGHT FIXTURE',
    'EXTERIOR LIGHT FIXTURE - HIGH GRADE',
    'EXTERIOR LIGHT FIXTURE - STANDARD GRADE',
    'EXTERIOR LIGHT FIXTURE - PREMIUM GRADE',
    'EXTERIOR POST LIGHT FIXTURE',
    'EXTERIOR POST LIGHT FIXTURE - HIGH GRADE',
    'EXTERIOR POST LIGHT FIXTURE - PREMIUM GRADE',
    'FIXTURE (CAN) FOR TRACK LIGHTING - HIGH GRADE',
    'FLOOR LEVELING CEMENT - AVERAGE',
    'FLOOR LEVELING CEMENT - LIGHT',
    'FLOOR LEVELING CEMENT - HEAVY',
    'FLOOR PREP (SCRAPE RUBBER BACK RESIDUE)',
    'FLOOR PREP/FALSH PATCH',
    'FLUORESCENT LIGHT FIXTURE',
    'FLUORESCENT LIGHT FIXTURE - HIGH GRADE',
    'FLUORESCENT LIGHT FIXTURE - STANDARD GRADE',
    'FLUORESCENT LIGHT FIXTURE LENS - WRAPAROUND (LENS ONLY)',
    "FLUORESCENT - ACOUSTIC GRID FIXTURE, 2' X 2'",
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - 2 TUBE - 2' HIGH GRD",
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - TWO TUBE, 2'X 4'",
    'FLUORESCENT - ACOUSTIC GRID FIXTURE - 4 TUBE - HIGH GRD',
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - FOUR TUBE, 2'X 4'",
    "FLUORESCENT - ONE TUBE - 2' - FIXTURE W/LENS",
    "FLUORESCENT - ONE TUBE - 4' - FIXTURE W/LENS",
    "FLUORESCENT - TWO TUBE - 4' - FIXTURE W/LENS",
    "FLUORESCENT - TWO TUBE - 6' - FIXTURE W/LENS",
    "FLUORESCENT - FOUR TUBE - 4' - FIXTURE W/LENS",
    "FLUORESCENT - FOUR TUBE - 8' - FIXTURE W/LENS",
    'GENERAL DEMOLITION',
    'GLUE DOWN CARPET',
    'GLUE DOWN CARPET - HEAVY TRAFFIC',
    'GLUE DOWN CARPET - HIGH GRADE',
    'GLUE DOWN CARPET - STANDARD GRADE',
    'GLUE DOWN CARPET - PREMIUM GRADE',
    'HANGING LIGHT FIXTURE',
    'HANGING LIGHT FIXTURE - HIGH GRADE',
    'HANGING LIGHT FIXTURE - PREMIUM GRADE',
    'HANGING DOUBLE LIGHT FIXTURE',
    'HEAT LAMP FIXTURE - ADJUSTABLE W/SHIELD',
    'LIGHT FIXTURE',
    'LIGHT FIXTURE - STANDARD GRADE',
    'LIGHT FIXTURE - PREMIUM GRADE',
    'LIGHT FIXTURE - HIGH GRADE',
    'LIGHT FIXTURE - WALL SCONCE',
    'LIGHT FIXTURE - WALL SCONCE - HIGH GRADE',
    'LIGHT FIXTURE (COVER ONLY) - SMALL SIZE',
    'LIGHT FIXTURE (COVER ONLY) - MEDIUM SIZE',
    'LIGHT FIXTURE (COVER ONLY) - LARGE SIZE',
    'LIGHT STRIP - INCANDESCENT',
    'MOTION SENSOR FOR EXTERIOR LIGHT FIXTURE',
    'PLASTIC CONTRACTOR DEBRIS BAG',
    'PORCELAIN LIGHT FIXTURE',
    'RECESSED LIGHT FIXTURE',
    'RECESSED LIGHT FIXTURE - STANDARD GRADE',
    'RECESSED LIGHT FIXTURE - HIGH GRADE',
    'RECESSED LIGHT FIXTURE - PREMIUM GRADE',
    'RECESSED LIGHT FIXTURE - TRIM ONLY',
    'RECESSED LIGHT FIXTURE - TRIM ONLY - HIGH GRADE',
    'ROUGH-IN FOR AUDIO/VISUAL SYSTEM (PER SF)',
    'ROUGH IN PLUMBING - INCLUDES SUPPLY AND WASTE LINES',
    'ROUGH IN PLUMBING - PER FIXTURE',
    "SCARF, SHAWL & WRAP - LADIES' - HIGH GRADE",
    'SCREW DOWN EXISTING SUBFLOOR - ELIMINATE FLOOR SQUEAKS',
    'SEAM CARPET',
    'SHEATHING - ADDITIONAL COST FOR H-CLIPS',
    'SHOWER LIGHT - WATERPROOF FIXTURE',
    'SPOT LIGHT FIXTURE - SINGLE',
    'SPOT LIGHT FIXTURE - DOUBLE',
    'SPOT LIGHT FIXTURE - DOUBLE - W/MOTION SENSOR',
    'TRACK LIGHTING - TRACK ONLY',
    'URETHANE FOAM SEALANT',
    'WRAP CUSTOM FASCIA WITH ALUMINUM (PER LF)', 
    'WRAP WOOD DOOR FRAME & TRIM WITH ALUMINUM (PER LF)',
    'WRAP WOOD GARAGE DOOR FRAME & TRIM WITH ALUMINUM (PER LF)',
    'WRAP WOOD POST WITH ALUMINUM (PER LF)',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - SMALL',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - LARGE',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - XLARGE'
]


STOP_CLAIM_IDS = [
    "tear off"
]

ZERO_WEIGHT_CLAIMS = [
    "concrete slab", "in place"
]

ALL_STOP_CLAIMS = STOP_CLAIMS + STOP_SMALL_ITEM_CLAIMS + STOP_DONATED_REPURPOSED_CLAIMS

KEEP_ITEMS = [
    "axle dump", 
    "dumpster load", 
    "pickup truck load", 
    "self adhesive"
]

PICKUP_TRUCK_WEIGHT = 0.5

TRUCK_COST_MAP = {
    "PICKUP TRUCK LOAD" : 234,
    "DUMPSTER LOAD" : {
        "TRUCK" : 234,
        "12 YARDS" : 358,
        "20 YARDS" : 471,
        "30 YARDS" : 930,
        "40 YARDS" : 1931,
    }
}

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


CLIENT_OUTPUT_RENAME_GENERAL = {
    'claim_id': 'Claim Number',
    'date': 'Date of Claim',
    'total_items': 'Number of items in claim needing collection'
}

CLIENT_OUTPUT_RENAME_XACTIMATE = {
    'total_truck_weight': 'Estimated weight of items needing collection (US Tons)',
    'truck_weights': 'Xactimate Waste Recommendation (Size - US Tons)',
    'truck_qty': 'Xactimate Waste Recommendation (Qty)',
    'total_truck_cost': 'Xactimate Estimated Cost for Waste Recommendation ($)'
}

CLIENT_OUTPUT_RENAME_PHOENIX = {
    'weight_estimation_ustons_median_median': 'Phoenix Estimated Weight (US Tons)',
    'estimated_truck_weights_median_median': 'Phoenix Waste Recommendation (Size - US Tons)',
    'estimated_truck_qty_median_median': 'Phoenix Waste recommendation (Qty)',
    'estimated_cost_median_median': 'Phoenix Estimated Cost for Waste Recommendation ($)'
}

CLIENT_OUTPUT_RENAME_SAVINGS = {
    'excessive_truck_weight_ustons_median_median': 'Weight Difference (US tons)',
    'excessive_cost_median_median': 'Cost Difference ($)',
    'relative_excessive_cost_median_median': 'Saving (%)'
}

CLIENT_OUTPUT_RENAME_MATCHING = {
    'total_id_matching_fraction': 'Fraction of matched claims (%)'
}

CLIENT_OUTPUT_RENAME_MAP = {
    **CLIENT_OUTPUT_RENAME_GENERAL, 
    **CLIENT_OUTPUT_RENAME_XACTIMATE, 
    **CLIENT_OUTPUT_RENAME_PHOENIX, 
    **CLIENT_OUTPUT_RENAME_SAVINGS, 
    **CLIENT_OUTPUT_RENAME_MATCHING
}

CLIENT_OUTPUT_COLORS = {
    "General" : "#C0C0C0",
    "Xactimate" : "#FF8080",
    "Phoenix" : "#00FFFF",
    "Savings" : "#339966",
    "Matching" : "yellow"
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

# DUMPSTER LOAD - APPROX. 12 YARDS, 1-3 TONS OF DEBRIS',
# DUMPSTER LOAD - APPROX. 30 YARDS, 5-7 TONS OF DEBRIS',
# DUMPSTER LOAD - APPROX. 40 YARDS, 7-8 TONS OF DEBRIS',
# DUMPSTER LOAD - APPROX. 20 YARDS, 4 TONS OF DEBRIS',
# HAUL DEBRIS - PER PICKUP TRUCK LOAD - INCLUDING DUMP FEES',
# SINGLE AXLE DUMP TRUCK - PER LOAD - INCLUDING DUMP FEES',
# TANDEM AXLE DUMP TRAILER - PER LOAD - INCLUDING DUMP FEES',
# LANDFILL FEES - (PER TON)'