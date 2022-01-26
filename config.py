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

ACTIVITIES = [
    'FLOOR LEVELING CEMENT - AVERAGE',
    'FLOOR LEVELING CEMENT - LIGHT',
    'FLOOR LEVELING CEMENT - HEAVY',
    'CONCRETE SEALER - BRUSH OR SPRAY APPLIED',
    'FLOOR PREP (SCRAPE RUBBER BACK RESIDUE)',
    'FLOOR PREPARATION FOR RESILIENT FLOORING',
    'FLOOR PREP/FALSH PATCH',
    'FLOOR PREPARATION FOR RESILIENT FLOORING - HEAVY',
    'GLUE DOWN CARPET',
    'GLUE DOWN CARPET - HEAVY TRAFFIC',
    'GLUE DOWN CARPET - HIGH GRADE',
    'GLUE DOWN CARPET - STANDARD GRADE',
    'GLUE DOWN CARPET - PREMIUM GRADE',
    'FLUORESCENT LIGHT FIXTURE',
    "FLUORESCENT - TWO TUBE - 4' - FIXTURE W/LENS",
    'LIGHT FIXTURE - STANDARD GRADE',
    'PORCELAIN LIGHT FIXTURE',
    'ROUGH IN PLUMBING - PER FIXTURE',
    'EXTERIOR LIGHT FIXTURE - PREMIUM GRADE',
    'SPOT LIGHT FIXTURE - DOUBLE - W/MOTION SENSOR',
    'LIGHT FIXTURE',
    'PLUMBING FIXTURE SUPPLY LINE',
    'RECESSED LIGHT FIXTURE',
    'HANGING LIGHT FIXTURE - PREMIUM GRADE',
    'LIGHT FIXTURE - PREMIUM GRADE',
    'LIGHT FIXTURES (BID ITEM)',
    'EXTERIOR LIGHT FIXTURE - HIGH GRADE',
    'EXTERIOR LIGHT FIXTURE',
    'EXTERIOR POST LIGHT FIXTURE',
    'LIGHT FIXTURE (COVER ONLY) - SMALL SIZE',
    'LIGHT FIXTURE - HIGH GRADE',
    'FLUORESCENT LIGHT FIXTURE LENS - WRAPAROUND (LENS ONLY)',
    'EXTERIOR POST LIGHT FIXTURE - HIGH GRADE',
    'RECESSED LIGHT FIXTURE - STANDARD GRADE',
    "FLUORESCENT - ACOUSTIC GRID FIXTURE, 2' X 2'",
    'SPOT LIGHT FIXTURE - DOUBLE',
    "FLUORESCENT - FOUR TUBE - 4' - FIXTURE W/LENS",
    'RECESSED LIGHT FIXTURE - TRIM ONLY',
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - FOUR TUBE, 2'X 4'",
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - TWO TUBE, 2'X 4'",
    'LIGHT FIXTURE (COVER ONLY) - LARGE SIZE',
    "FLUORESCENT - ONE TUBE - 4' - FIXTURE W/LENS",
    "FLUORESCENT - ACOUSTIC GRID FIXTURE - 2 TUBE - 2' HIGH GRD",
    'RECESSED LIGHT FIXTURE - HIGH GRADE',
    "FLUORESCENT - ONE TUBE - 2' - FIXTURE W/LENS",
    'FLUORESCENT LIGHT FIXTURE - STANDARD GRADE',
    'LIGHT FIXTURE (COVER ONLY) - MEDIUM SIZE',
    'SPOT LIGHT FIXTURE - SINGLE',
    'HANGING LIGHT FIXTURE',
    'LIGHT FIXTURE - WALL SCONCE',
    'MOTION SENSOR FOR EXTERIOR LIGHT FIXTURE',
    'FLUORESCENT - ACOUSTIC GRID FIXTURE - 4 TUBE - HIGH GRD',
    'RECESSED LIGHT FIXTURE - PREMIUM GRADE',
    'EXTERIOR LIGHT FIXTURE - STANDARD GRADE',
    'RECESSED LIGHT FIXTURE - TRIM ONLY - HIGH GRADE',
    'HANGING DOUBLE LIGHT FIXTURE',
    "FLUORESCENT - FOUR TUBE - 8' - FIXTURE W/LENS",
    'FLUORESCENT LIGHT FIXTURE - HIGH GRADE',
    'HANGING LIGHT FIXTURE - HIGH GRADE',
    'SHOWER LIGHT - WATERPROOF FIXTURE',
    "FLUORESCENT - TWO TUBE - 6' - FIXTURE W/LENS",
    'LIGHT FIXTURE - WALL SCONCE - HIGH GRADE',
    'HEAT LAMP FIXTURE - ADJUSTABLE W/SHIELD',
    'EXTERIOR POST LIGHT FIXTURE - PREMIUM GRADE',
    'FIXTURE (CAN) FOR TRACK LIGHTING - HIGH GRADE',
    'DUMPSTER LOAD - APPROX. 12 YARDS, 1-3 TONS OF DEBRIS',
    'DUMPSTER LOAD - APPROX. 30 YARDS, 5-7 TONS OF DEBRIS',
    'DUMPSTER LOAD - APPROX. 40 YARDS, 7-8 TONS OF DEBRIS',
    'HAUL DEBRIS - PER PICKUP TRUCK LOAD - INCLUDING DUMP FEES',
    'GENERAL DEMOLITION (BID ITEM)',
    'DUMPSTER LOAD - APPROX. 20 YARDS, 4 TONS OF DEBRIS',
    'GENERAL DEMOLITION',
    'PLASTIC CONTRACTOR DEBRIS BAG',
    'TEAR OFF, HAUL AND DISPOSE OF COMP. SHINGLES - LAMINATED',
    'TEAR OFF, HAUL AND DISPOSE OF COMP. SHINGLES - 3 TAB',
    'TEAR OFF, HAUL AND DISPOSE OF GRAVEL BALLAST',
    'TEAR OFF, HAUL AND DISPOSE OF MODIFIED BITUMEN ROOFING',
    'TEAR OFF, HAUL AND DISPOSE OF WOOD SHAKES/SHINGLES',
    'TEAR OFF, HAUL AND DISPOSE OF 4 PLY BUILT-UP ROOFING',
    'TEAR OFF, HAUL AND DISPOSE MEMBRANE ROOFING - FULL ADHERED',
    'TEAR OFF, HAUL AND DISPOSE OF STEEL TILE',
    'TEAR OFF, HAUL AND DISPOSE OF 3 PLY BUILT-UP ROOFING',
    'TEAR OFF, HAUL AND DISPOSE OF COMP. SHINGLES - HIGH PRO.',
    'TEAR OFF, HAUL AND DISPOSE CORRUGATED FIBERGLASS ROOFING',
    'TEAR OFF, HAUL AND DISPOSE MEMBRANE ROOFING - PER. ADHERED',
    'TEAR OFF, HAUL AND DISPOSE OF ROLL ROOFING',
    'TEAR OFF, HAUL AND DISPOSE OF 5 PLY BUILT-UP ROOFING',
    'TEAR OFF, HAUL AND DISPOSE OF TILE ROOFING',
    'TEAR OFF, HAUL AND DISPOSE RUBBER ROOFING - PER ADHERED',
    'TEAR OFF, HAUL AND DISPOSE OF SYNTHETIC COMPOSITE ROOFING',
    'TEAR OFF, HAUL AND DISPOSE OF HIGH GRADE ROLL ROOFING',
    '5/8" DRYWALL - HUNG, TAPED, READY FOR TEXTURE',
    'TEXTURE DRYWALL - LIGHT HAND TEXTURE',
    'TEXTURE DRYWALL - HEAVY HAND TEXTURE',
    '1/2" DRYWALL - HUNG, TAPED, READY FOR TEXTURE',
    'ACOUSTIC CEILING (POPCORN) TEXTURE',
    'TEXTURE DRYWALL - MACHINE',
    '1/2" WATER ROCK (GREENBOARD) HUNG, TAPED READY FOR TEXTURE',
    'TEXTURE DRYWALL - MACHINE - KNOCKDOWN',
    'TEAR OFF PAINTED ACOUSTIC CEILING (POPCORN) TEXTURE',
    '1/2" MOLD/MILDEW RESISTANT - HUNG, TAPED READY FOR TEXTURE',
    'SCRAPE OFF ASBESTOS ACOUSTIC (POPCORN) TEXTURE-NO HAUL OFF',
    '5/8" DRYWALL - TYPE C - HUNG, TAPED, LIGHT TEXTURE',
    'ACOUSTIC CEILING (POPCORN) TEXTURE - HEAVY',
    '1/2" ACOUSTIC DRYWALL - HUNG, TAPED, READY FOR TEXTURE',
    'ACOUSTIC CEILING (POPCORN) TEXTURE - LIGHT',
    'ROOFING (BID ITEM)',
    'PLUMBING (BID ITEM)',
    'HEAT, VENT, & AIR CONDITIONING (BID ITEM)',
    'ELECTRICAL (BID ITEM)',
    'DRYWALL (BID ITEM)',
    'DOORS (BID ITEM)',
    'ELECTRICAL - SPECIAL SYSTEMS (BID ITEM)',
    'FENCING (BID ITEM)',
    'SOFFIT, FASCIA, & GUTTER (BID ITEM)',
    'GLASS, GLAZING, & STOREFRONTS (BID ITEM)',
    'CABINETRY (BID ITEM)',
    'FRAMING & ROUGH CARPENTRY (BID ITEM)',
    'SPECIALTY ITEMS (BID ITEM)',
    'APPLIANCES (BID ITEM)',
    'LANDSCAPING (BID ITEM)',
    'RENTAL EQUIPMENT DELIVERY / MOBILIZATION (BID ITEM)',
    'CONCRETE & ASPHALT (BID ITEM)',
    'SIDING (BID ITEM)',
    'MASONRY (BID ITEM)',
    'HEAVY EQUIPMENT (BID ITEM)',
    'EXTERIOR STRUCTURES (BID ITEM)',
    'INSULATION (BID ITEM)',
    'WINDOWS - VINYL (BID ITEM)',
    'WINDOWS - SKYLIGHTS (BID ITEM)',
    'TUNNELING UNDER SLAB (BID ITEM)',
    'MIRRORS & SHOWER DOORS (BID ITEM)',
    'STABILIZATION: CONCRETE PUSH PIER SYSTEM (BID ITEM)',
    'SCAFFOLDING (BID ITEM)',
    'BID ITEM',
    'A/C FIN CONDENSER CAGE (BID ITEM)',
    'MARBLE - CULTURED / NATURAL (BID ITEM)',
    'PANELING & WOOD WALL FINISHES (BID ITEM)',
    'ROCK WALL / LANDSCAPING BOULDERS - (BID ITEM)', 
    'DUMPSTER LOAD - APPROX. 12 YARDS, 1-3 TONS OF DEBRIS',
    'DUMPSTER LOAD - APPROX. 30 YARDS, 5-7 TONS OF DEBRIS',
    'DUMPSTER LOAD - APPROX. 40 YARDS, 7-8 TONS OF DEBRIS',
    'HAUL DEBRIS - PER PICKUP TRUCK LOAD - INCLUDING DUMP FEES',
    'GENERAL DEMOLITION (BID ITEM)',
    'DUMPSTER LOAD - APPROX. 20 YARDS, 4 TONS OF DEBRIS',
    'GENERAL DEMOLITION',
    'PLASTIC CONTRACTOR DEBRIS BAG',
    'P-TRAP ASSEMBLY - ABS (PLASTIC)',
    'STORM DOOR ASSEMBLY',
    'SINK STRAINER AND DRAIN ASSEMBLY',
    'STORM DOOR ASSEMBLY - HIGH GRADE',
    'STORM DOOR ASSEMBLY - PREMIUM GRADE',
    'P-TRAP ASSEMBLY - ACID WASTE PIPING (PVDF)',
    'SINK STRAINER AND DRAIN ASSEMBLY - HIGH GRADE',
    'SINK DRAIN ASSEMBLY WITH STOP',
    'STORM DOOR ASSEMBLY - STANDARD GRADE',
    'OPEN AND CLOSE SLAB FOR PLUMBING WORK',
    'MASK WALL - PLASTIC, PAPER, TAPE (PER LF)',
    'MASK PER SQUARE FOOT FOR DRYWALL WORK',
    'TEAR OFF ASBESTOS DRYWALL (NO HAUL OFF)',
    'TEAR OFF MODIFIED BITUMEN ROOFING (NO HAUL OFF)',
    'TEAR OFF 3 PLY BUILT-UP ROOFING (NO HAUL OFF)',
    'TEAR OFF ASBESTOS ACOUSTIC CEILING TILE (NO HAUL OFF)',
    'TEAR OFF ASBESTOS SIDING (NO HAUL OFF)',
    'TEAR OFF WOOD SHAKES/SHINGLES (NO HAUL OFF)',
    'TEAR OFF RUBBER ROOFING - PER ADHERED (NO HAUL OFF)',
    'TEAR OFF GRAVEL BALLAST (NO HAUL OFF)',
    'TEAR OFF COMPOSITION SHINGLES (NO HAUL OFF)',
    'TEAR OFF MEMBRANE ROOFING - PERIM. ADHERED (NO HAUL OFF)',
    'TEAR OFF MEMBRANE ROOFING - FULLY ADHERED (NO HAUL OFF)',
    'TEAR OFF RIGID ASBESTOS SHINGLES (NO HAUL OFF)',
    'FILL HOLES CREATED BY WALL CAVITY DRYING', 
    'DRYWALL TAPE JOINT/REPAIR - PER LF',
    'TAPE JOINT FOR NEW TO EXISTING DRYWALL - PER LF',
    'WRAP CUSTOM FASCIA WITH ALUMINUM (PER LF)', 
    'WRAP WOOD DOOR FRAME & TRIM WITH ALUMINUM (PER LF)',
    'WRAP WOOD GARAGE DOOR FRAME & TRIM WITH ALUMINUM (PER LF)',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - SMALL',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - LARGE',
    'WRAP WOOD WINDOW FRAME & TRIM WITH ALUMINUM SHEET - XLARGE',
    'WRAP WOOD POST WITH ALUMINUM (PER LF)',
    "SCARF, SHAWL & WRAP - LADIES' - HIGH GRADE"
]

STOP_CLAIM_IDS = [
    "bid item", "tear off haul dispose"
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

