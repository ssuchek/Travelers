import pandas as pd
import re
import time

from config import config
import config as constants

from statistics import median, mean, mode

from utils.loader.loader import ClaimDataLoader    

from utils.preprocess import Preprocessor, PreprocessTransformation
from utils.preprocess import BASIC_PREPROCESS, CATEGORIES_WORD_PREPROCESS
from utils.preprocess import WEIGHTS_PREPROCESS, WEIGHTS_WORD_PREPROCESS

from utils.helpers import format_and_regex

version = config["data"]["db_version"]
output_path = config["data"]["output_dir"]

weights = pd.read_csv(config["data"]["weights_preprocessed_db"])

data = pd.read_csv("{0}/{1}/claims_weight_db_matched_{1}.csv".format(output_path, version))

from statistics import median, mean, mode
import time

def calculate_units(weights, data, id_col="pentatonic_id", primary_col="primary_desc", weight_col="weight_ustons", zero_claims=constants.ACTIVITIES, remove_ids=constants.STOP_CLAIM_IDS, **kwargs):

    print("Unit matching...")
    data["unit_matching"] = "undefined"
    valid_unit_mask = (data["unit"].notna() & (data["unit"] != ""))
    unit_matching_mask = (data["unit"] == data["item_unit_cd"])
    data.loc[unit_matching_mask & valid_unit_mask, "unit_matching"] = "yes"
    data.loc[~unit_matching_mask & valid_unit_mask, "unit_matching"] = "no"
    
    primary_methods = kwargs.get("primary_methods")
    
    methods = kwargs.get("methods")
    
    new_cols = []

    if remove_ids:
        remove_id_regex = format_and_regex(";".join(remove_ids))
        print("Removing IDs with following items: {}".format(remove_id_regex))

        print("Word preprocessing...")
        word_preprocessor = Preprocessor(CATEGORIES_WORD_PREPROCESS)
        data = word_preprocessor.calculate(data)

        total_claim_ids = len(data["claim_id"].dropna().unique().tolist())

        remove_mask = data["item_description_processed"].astype(str).str.contains(remove_id_regex, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)
        remove_ids_list = data.loc[remove_mask, "claim_id"].dropna().unique().tolist()
        
        print("Total {}/{} claims from {} claim IDs contain remove ID patterns".format(remove_mask.sum(),
                                                                                    data.shape[0],
                                                                                    len(remove_ids_list)
                                                                                ))

        data = data[~data["claim_id"].isin(remove_ids_list)]

        print("Total {}/{} claim IDs are removed as containing remove ID claims".format(total_claim_ids-len(data["claim_id"].dropna().unique().tolist()),
                                                                                    total_claim_ids
                                                                                ))
    
    for (method, primary_method) in zip(methods, primary_methods):
        
        if not method:
            method = "zero" 
        
        if not primary_method:
            primary_method = "zero" 
        
        new_cols.append(weight_col + "_" + method + "_" + primary_method)
    
    data[new_cols] = 0
    
    if not methods and not primary_methods:
        return data

    
    
    data = data.merge(weights[[id_col, weight_col]], how="left", on=id_col)

    def calculate(row, unit_col, id_col, reference_df, **kwargs):
        
        method = kwargs.get("method")
        
        if not method:
            return 0
        
        primary_col = kwargs.get("primary_col")
        
        if not primary_col:
            ids = [x.strip() for x in row[id_col].split(',')]
        else:
            primary_values = [p.strip() for p in row["weights_" + primary_col].split(',')]
            reference_mask = reference_df[primary_col].isin(primary_values) & (reference_df["unit"] == row["item_unit_cd"])
            ids = reference_df.loc[reference_mask, id_col].dropna().unique().tolist()
         
        units = reference_df.loc[reference_df[id_col].isin(ids), unit_col].dropna().unique().tolist()
        
        if not units:
            return 0
          
        if method == "max":
            return max(units)
        elif method == "mean":
            return mean(units)
        elif method == "min":
            return min(units)
        elif method == "median":
            return median(units)
        
    if not isinstance(methods, list):
        methods = [methods]
        
    if not isinstance(primary_methods, list):
        primary_methods = [primary_methods]
        
    # Claims with valid matched ID field
    valid_id_mask = (data[id_col].notna() & (data[id_col] != ""))

    # Claims that are marked to have zero weight (activities)
    zero_weight_mask = data["item_description"].isin(zero_claims) if zero_claims else True

    print("Total {}/{} claims are marked as having zero weight".format(zero_weight_mask.sum(),
                                                                        data.shape[0]
                                                                        ))
    # Claims with valid matched primary desciption
    valid_primary_mask = (data["weights_" + primary_col].notna() & (data["weights_" + primary_col] != ""))

    # Claims that have multiple IDs associated 
    calculate_id_mask = data[id_col].fillna("").str.contains(",")
       
    for (new_col, method, primary_method) in zip(new_cols, methods, primary_methods):

        single_id_mask = valid_id_mask  & ~calculate_id_mask & ~zero_weight_mask
        data.loc[single_id_mask, new_col]  = data.loc[single_id_mask, weight_col]

        multiple_id_mask = valid_id_mask & calculate_id_mask & ~zero_weight_mask
        data.loc[multiple_id_mask, new_col] = data[multiple_id_mask].apply(lambda row: calculate(row, weight_col, id_col, weights, method=method), axis=1)

        umatched_id_matched_primary = ~valid_id_mask & valid_primary_mask & ~zero_weight_mask
        data.loc[umatched_id_matched_primary, new_col] = data[umatched_id_matched_primary].apply(lambda row: calculate(row, weight_col, id_col, weights, method=primary_method, primary_col=primary_col), axis=1)
    
    data = data.drop(columns=[weight_col])
    
    return data

loader = ClaimDataLoader()
    
methods = ["max", "mean", "median", "min"]

id_methods = methods + methods
primary_methods = methods + [""]*len(methods)

print("Start calculating weights...")
start_time = time.time()  

weights_data = calculate_units(weights=weights, data=data.copy(), methods=id_methods, primary_methods=primary_methods) 

print("Time elapsed: {} s".format(time.time()-start_time))

filename = "{0}/{1}/claims_weight_db_matched_weights_{1}.csv".format(output_path, version)

loader.save_data_to_csv(weights_data, filename, index=False)
