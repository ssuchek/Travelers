import collections
import ijson
import itertools
import logging
import os
import re
import requests
import shutil
import statistics

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import config as constants
from config import config

from utils.preprocess import Preprocessor, PreprocessTransformation
from utils.preprocess import BASIC_PREPROCESS, CATEGORIES_WORD_PREPROCESS
from utils.preprocess import WEIGHTS_PREPROCESS, WEIGHTS_WORD_PREPROCESS, WEIGHTS_WORD_FREQUENCY_PREPROCESS
from utils.preprocess.dataframe import aggregate_col_values_to_comma_list

from utils.logging.helpers import log_and_warn
from utils.helpers import format_and_regex


def read_if_exist_decorator(get_data):
    def _read_file_if_exists(*args, **kwargs):
        try:
            filename = kwargs["filename"]
            if os.path.exists(filename):
                logging.info("Loading from file {}...".format(filename))
                try:
                    data = pd.read_csv(filename, na_filter=False)
                    logging.info("Data loaded from file {}".format(filename))
                except Exception:
                    try:
                        with open(filename, 'rb') as f:
                            items = list(ijson.items(f, '', use_float=True))
                        if len(items) > 0:
                            data = items[0]
                        logging.info("Data loaded from file {}".format(filename))
                    except Exception:
                        try:
                            data = pd.read_excel(filename, engine="openpyxl")
                            logging.info("Data loaded from file {}".format(filename))
                        except Exception as e:
                            logging.info("Failed to load {} due to exception: {}".format(filename, e))
                            raise Exception(e)
            else:
                data = None
                logging.info("File {} not found. No data loaded".format(filename))

            kwargs["filename"] = filename
            return get_data(data=data, *args, **kwargs)
        except Exception as e:
            raise Exception(e)

    return _read_file_if_exists


class ClaimDataLoader(object):

    def __init__(self, url=None, db_connector=None, auth=None):
        self.url = url
        self.auth = auth
        self.session = self.create_session(retries=3)
        self.db_connector = db_connector

    def create_session(self, retries=3):
        """
        Create session object with auth and specified number of retries
        :param retries: allowed number of retries
        :return: session object
        """
        session = requests.Session()
        if self.auth:
            session.auth = self.auth
        retries = Retry(total=retries, backoff_factor=0.2, status_forcelist=[500])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def single_api_call(self, url, params=None, timeout=600, filename=None):
        """
        Single get request with specified timeout
        :return: responce as json object with data and its time if any was received
        """
        if not params:
            params = {}

        try:
            with self.session.get(url, params=params, timeout=timeout, stream=True) as r:
                r.raise_for_status()
                with open(filename, "wb") as f:
                    shutil.copyfileobj(r.raw, f, length=16 * 1024 * 1024)
            logging.info("Downloaded: {}".format(filename))
            responce_time = r.elapsed.total_seconds()
        except Exception as e:
            msg = "Exception on API call to tennis-data: {}".format(type(e))
            logging.error(msg)
            raise Exception(msg)

    def save_data_to_csv(self, data, filename, **kwargs):
        """
            Save df to csv
            :param data: df to save
            :param filename: path to file
            :param kwargs: additional arguments for DataFrame.to_csv method
        """
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        data.to_csv(filename, **kwargs)

    def save_data_to_excel(self, data, filename, **kwargs):
        """
            Save df to csv
            :param data: df to save
            :param filename: path to file
            :param kwargs: additional arguments for DataFrame.to_csv method
        """
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        data.to_excel(filename, **kwargs)

    def save_data_to_json(self, data, filename, **kwargs):
        """
            Save df to csv
            :param data: df to save
            :param filename: path to file
            :param kwargs: additional arguments for DataFrame.to_csv method
        """
        dirname = os.path.dirname(filename)
        os.makedirs(dirname, exist_ok=True)
        data.to_json(filename, **kwargs)

    @read_if_exist_decorator
    def load_claim_data(self, filename, input_files, data=None):

        if data is None:
            file_data = []

            for file in input_files:
                logging.info("Loading file {}".format(file))
                try:
                    file_data.append(pd.read_excel(file))
                except Exception as e:
                    logging.info("Input Excel file is not readable: {}".format(e))

            data = pd.concat(file_data)

            logging.info("Total number of claims: {}".format(data.shape[0]))

            data = data.rename(columns=constants.FIELD_RENAME_MAP)

            self.save_data_to_csv(data, filename, index=False)

        return data

    @read_if_exist_decorator
    def preprocess_claims(self, claims, filename, data=None):
        logging.info("Start preprocessing claims")
        if data is None:
            data = claims.copy()

            logging.info("Basic data preprocessing...")
            basic_preprocessor = Preprocessor(BASIC_PREPROCESS)
            data = basic_preprocessor.calculate(data)

            logging.info("Word preprocessing...")
            word_preprocessor = Preprocessor(CATEGORIES_WORD_PREPROCESS)
            data = word_preprocessor.calculate(data)

            total_claims_start = data.shape[0]

            regex_keep_claims = ";".join(constants.KEEP_ITEMS)

            regex_keep_claims_expr = format_and_regex(regex_keep_claims)

            for stop_claim in constants.STOP_CLAIM_IDS:
                logging.info("Processing stop claim ID {}...".format(stop_claim))

                regex_stop_claims_expr = format_and_regex(stop_claim)

                logging.info("Regex pattern: {}".format(regex_stop_claims_expr))

                stop_claims_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                                    data["item_description_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                        data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)  

                stop_claims_mask &= ~keep_mask   

                log_and_warn(
                        "Total {}/{} claims are filtered out according to {}".format(stop_claims_mask.sum(),
                                                                                        data.shape[0],
                                                                                        stop_claim
                                                                                    ))

                roofing_mask = data["subcategory_prev_processed"].str.contains("roofing", 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)
    
                stop_claim_ids = data.loc[stop_claims_mask & roofing_mask, "claim_id"].unique().tolist()
                remove_mask = data["claim_id"].isin(stop_claim_ids) & roofing_mask

                log_and_warn("Total {}/{} roofing claims are filtered out according to {}".format(remove_mask.sum(),
                                                                                                    data.shape[0],
                                                                                                    stop_claim
                                                                                                ))

                if remove_mask.sum() > 0:
                    data = data[~remove_mask]

            for stop_claim in constants.STOP_CLAIMS_CONTAINING:
                logging.info("Processing stop claim {}...".format(stop_claim))

                regex_stop_claims_expr = format_and_regex(stop_claim, special_delimiters=None)

                logging.info("Regex pattern: {}".format(regex_stop_claims_expr))

                stop_claims_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                                    data["item_description_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                            data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                stop_claims_mask &= ~keep_mask   

                log_and_warn(
                        "Total {}/{} claims are filtered out according to {}".format(stop_claims_mask.sum(),
                                                                                        data.shape[0],
                                                                                        stop_claim
                                                                                    ))

                if stop_claims_mask.sum() > 0:
                    data = data[~stop_claims_mask]

            for stop_claim in constants.ALL_STOP_CLAIMS:
                logging.info("Processing stop claim {}...".format(stop_claim))

                regex_stop_claims_expr = format_and_regex(stop_claim)

                logging.info("Regex pattern: {}".format(regex_stop_claims_expr))

                stop_claims_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                                    data["item_description_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                            data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)   

                stop_claims_mask &= ~keep_mask   

                log_and_warn(
                        "Total {}/{} claims are filtered out according to {}".format(stop_claims_mask.sum(),
                                                                                        data.shape[0],
                                                                                        stop_claim
                                                                                    ))

                if stop_claims_mask.sum() > 0:
                    data = data[~stop_claims_mask]

            for stop_claim in constants.STOP_CATEGORIES:
                logging.info("Processing stop category {}...".format(stop_claim))

                regex_stop_claims_expr = format_and_regex(stop_claim)

                logging.info("Regex pattern: {}".format(regex_stop_claims_expr))

                stop_claims_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                                    data["item_description_processed"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                        data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)        
                stop_claims_mask &= ~keep_mask       

                log_and_warn(
                        "Total {}/{} claims are filtered out according to {}".format(stop_claims_mask.sum(),
                                                                                        data.shape[0],
                                                                                        stop_claim
                                                                                    ))

                if stop_claims_mask.sum() > 0:
                    data = data[~stop_claims_mask]

            for stop_claim in constants.STOP_REASON_DESC:
                logging.info("Processing stop category {}...".format(stop_claim))

                regex_stop_claims_expr = format_and_regex(stop_claim)

                logging.info("Regex pattern: {}".format(regex_stop_claims_expr))

                stop_claims_mask = data["primary_col_desc"].astype(str).str.contains(regex_stop_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)

                keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                        data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)        
                stop_claims_mask &= ~keep_mask

                log_and_warn(
                        "Total {}/{} claims are filtered out according to {}".format(stop_claims_mask.sum(),
                                                                                        data.shape[0],
                                                                                        stop_claim
                                                                                    ))


                if stop_claims_mask.sum() > 0:
                    data = data[~stop_claims_mask]   

            logging.info("Processing stop activities...")

            stop_activities_mask = data["item_description_processed"].isin(constants.STOP_ACTIVITIES)

            keep_mask = data["subcategory_prev_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                        data["item_description_processed"].astype(str).str.contains(regex_keep_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)  

            stop_activities_mask &= ~keep_mask

            log_and_warn(
                        "Total {}/{} claims are filtered out according to stop activities".format(stop_activities_mask.sum(),
                                                                                        data.shape[0]
                                                                                    ))

            if stop_activities_mask.sum() > 0:
                data = data[~stop_activities_mask]                                                                        
    
            log_and_warn(
                        "Total {}/{} claims are filtered out according to all stop categories, activites and claims".format(total_claims_start-data.shape[0],
                                                                                        total_claims_start
                                                                                    ))

            data = self.match_zip_codes(data)

            drop_columns = [col for col in data.columns if col.endswith("_processed")]
            if drop_columns:
                data = data.drop(columns=drop_columns)

            self.save_data_to_csv(data, filename, index=False)

        return data

    @read_if_exist_decorator
    def preprocess_weights_db(self, weights_db_file, filename, data=None):
        logging.info("Start preprocessing weights DB")
        if data is None:
            data = pd.read_csv(weights_db_file)

            data = data[constants.WEIGHTS_DB_FIELDS]

            logging.info("Start basic preprocessing of weights DB")
            basic_preprocessor = Preprocessor(WEIGHTS_PREPROCESS)
            data = basic_preprocessor.calculate(data)

            self.save_data_to_csv(data, filename, index=False)

        return data

    @read_if_exist_decorator
    def get_word_frequency(self, claims, filename, column, words=None, data=None, transformations=WEIGHTS_WORD_FREQUENCY_PREPROCESS):

        if data is None:
            data = claims.copy()

            if transformations:
                frequency_preprocessor = Preprocessor(transformations)
                data = frequency_preprocessor.calculate(data)

            if words:
                if isinstance(words, list):
                    regex_expr = format_and_regex(";".join(words))        
                else:
                    regex_expr = format_and_regex(words)   

                words_mask = data[column].astype(str).str.contains(regex_expr, flags=re.IGNORECASE, regex=True)
                data = data[words_mask]
                
                logging.info("Start frequency analysis for claims containing words: {}".format(words))
            else:
                logging.info("Start frequency analysis for all claims")

            data[column] = data[column].astype(str).str.split()

            nested_item_list = data[column].values.tolist()

            item_list = list(itertools.chain.from_iterable(nested_item_list))

            words = collections.Counter(item_list)

            total_words = len(set(words))

            logging.info("Total unique words: {}".format(total_words))

            word_frequency = pd.DataFrame(words.most_common(total_words),
                                            columns=['words', 'count'])


            self.save_data_to_csv(word_frequency, filename, index=False)
            self.save_data_to_excel(word_frequency, filename.replace(".csv", ".xlsx"), index=False)

            return word_frequency

        return data

    def plot_get_word_frequency(self, data, filename):

        fig, (ax1, ax2) = plt.subplots(1, 2)

        plt.subplots_adjust(left=1.6, right=3.2)

        fig1 = data.head(50).sort_values(by='count').plot(kind='barh',
                                                          x='words',
                                                          y='count',
                                                          ax=ax1,
                                                          color="red",
                                                          figsize=(8, 20))

        ax1.set_title("Fifty most common words in item description")

        fig2 = data.head(100).sort_values(by='count').plot(kind='barh',
                                                           x='words',
                                                           y='count',
                                                           ax=ax2,
                                                           color="purple",
                                                           figsize=(8, 20))

        ax2.set_title("Hundred most common words in item description")

        plt.savefig(filename, bbox_inches='tight', facecolor='w', dpi=600)

    def add_tag(self, data, mask, tag_col, tag, logging=True):
        """
        Aggregates multiple tags in the single value of type tag1,tag2,tag3,...
        :param data:           an input DataFrame
        :param mask:            an input DataFrame mask     
        :param tag_col:         name of the tag column
        :param tag:             an input tag to incorporate in tag column
        """
        if not isinstance(tag, str):
            tag = str(tag)

        tag_mask = data[tag_col].fillna("").astype(str).str.contains(tag + ",") | data[tag_col].fillna("").astype(str).str.contains(
            "," + tag) | (data[tag_col] == tag)

        if logging:
            log_and_warn("{}/{} claims are previously tagged as {}: no need to retag".format((mask & tag_mask).sum(),
                                                                                        data.shape[0],
                                                                                        tag
                                                                                        ))

        def add_tag_to_str(x, tag):

            if not isinstance(tag, str):
                tag = str(tag)

            if not isinstance(x, str):
                x = str(x)

            if x == "" or x == 0:
                return tag

            if not tag + "," in x and not "," + tag in x:
                x = x + "," + tag

            return x

        if (mask & (~tag_mask)).sum() > 0:
            data.loc[mask & (~tag_mask), tag_col] = data.loc[mask & (~tag_mask), tag_col].fillna("").apply(
                lambda x: add_tag_to_str(x, tag))

        if logging:
            log_and_warn("Tagged {}/{} claims as {}".format((mask & (~tag_mask)).sum(),
                                                       data.shape[0],
                                                       tag
                                                       ))

        return data

    def replace_tag(self, data, mask, tag_col, tag, logging=True):
        """
        Aggregates multiple tags in the single value of type tag1,tag2,tag3,...
        :param data:           an input DataFrame
        :param mask:            an input DataFrame mask     
        :param tag_col:         name of the tag column
        :param tag:             an input tag to incorporate in tag column
        """
        
        if logging:
            log_and_warn("For {}/{} claims tag in column {} is replaced with value {}".format(mask.sum(),
                                                                        data.shape[0],
                                                                        tag_col,
                                                                        tag
                                                                        ))

        data.loc[mask, tag_col] = tag

        return data

    @read_if_exist_decorator
    def calculate_categories(self, claims, categories, filename, data=None):

        if data is None:
            data = claims.copy()

            logging.info("Word preprocessing...")
            word_preprocessor = Preprocessor(CATEGORIES_WORD_PREPROCESS)
            data = word_preprocessor.calculate(data)

            data["category"] = ""
            data["subcategory"] = ""

            item_desc = data["item_description_processed"].dropna().unique().tolist()
            total_item_desc = len(item_desc)

            if total_item_desc > 0:
                logging.info("Found {} unique item descriptions in claims DB".format(total_item_desc))
            else:
                logging.error("Found no item descriptions in claims DB!".format(total_item_desc))
                raise Exception("Weights DB matching failed due to item descriptions not found in claims DB. Check your claim data")

            subcategory_desc = data["subcategory_prev_processed"].dropna().unique().tolist()
            total_subcategory_desc = len(subcategory_desc)

            if total_subcategory_desc > 0:
                logging.info("Found {} unique subcategories in claims DB".format(total_subcategory_desc))
            else:
                logging.error("Found no subcategories in claims DB!".format(total_subcategory_desc))
                raise Exception("Weights DB matching failed due to subcategories not found in claims DB. Check your claim data")

            for category, subcategory_map in categories.items():
                logging.info("Processing category {}...".format(category))
                for subcategory, patterns in subcategory_map.items():
                    logging.info("Processing subcategory {}...".format(subcategory))
                    logging.info("Search patterns: {}".format(patterns))

                    matched_item_desc = []
                    matched_subcategory_desc = []

                    for pattern in patterns:

                        compiled_regex_desc = re.compile(format_and_regex(pattern, permutations=True, is_synonyms=False))

                        logging.info("Regex pattern: {}".format(compiled_regex_desc))

                        matched_item_desc.extend(list(filter(compiled_regex_desc.match, item_desc)))
                        matched_subcategory_desc.extend(list(filter(compiled_regex_desc.match, subcategory_desc)))

                    total_matched_item_desc = len(matched_item_desc)    
                    total_matched_subcategory_desc = len(matched_item_desc)
                        
                    if total_matched_item_desc > 0 or total_matched_subcategory_desc > 0:
                        regex_mask = data["subcategory_prev_processed"].isin(matched_subcategory_desc) | \
                                    data["item_description_processed"].isin(matched_item_desc)
                        data = self.add_tag(data, regex_mask, "subcategory", subcategory)

                    subcategory_mask = data["subcategory"].astype(str).str.contains(subcategory, flags=re.IGNORECASE, regex=True)

                    log_and_warn("Total {}/{} items belong to subcategory {}".format(subcategory_mask.sum(),
                                                                                    data.shape[0],
                                                                                    subcategory
                                                                                    ))

                    data = self.add_tag(data, subcategory_mask, "category", category)

                category_mask = data["category"].astype(str).str.contains(category, flags=re.IGNORECASE, regex=True)
                log_and_warn("Total {}/{} items belong to category {}".format(category_mask.sum(),
                                                                              data.shape[0],
                                                                              category
                                                                              ))

            drop_columns = [col for col in data.columns if col.endswith("_processed")]
            if drop_columns:
                data = data.drop(columns=drop_columns)

            self.save_data_to_csv(data, filename, index=False)

        return data

    def calculate_categories_stats(self, data, categories, filename, states=["TX"]):

        if not os.path.exists(filename):
            categories_stats = {}
            categories_stats["category"] = {}

            for category, subcategory_map in categories.items():
                categories_stats["category"][category] = {}
                categories_stats["category"][category]["subcategories"] = {}
                logging.info("Calculating stats for category {}...".format(category))
                cat_mask = data["category"].fillna("").str.contains(category + ",") | \
                    data["category"].fillna("").str.contains("," + category) | (data["category"] == category)
                if cat_mask.sum() > 0:
                    log_and_warn("Total {}/{} claims belong to {} category".format(cat_mask.sum(),
                                                                                    data.shape[0],
                                                                                    category
                                                                                ))
                    categories_stats["category"][category]["totalclaims"] = int(cat_mask.sum())
                    categories_stats["category"][category]["totalitems"]  = int(data.loc[cat_mask, "item_quantity"].sum())
                else:
                    log_and_warn("No claims belong to {} category".format(category))
                for subcategory, _ in subcategory_map.items():
                    categories_stats["category"][category]["subcategories"][subcategory] = {}
                    logging.info("Calculating stats for subcategory {}...".format(subcategory))
                    subcat_mask = data["subcategory"].fillna("").str.contains(subcategory + ",") | data[
                        "subcategory"].fillna("").str.contains("," + subcategory) | (data["subcategory"] == subcategory)
                    if cat_mask.sum() > 0:
                        log_and_warn("Total {}/{} claims belong to {} subcategory".format(subcat_mask.sum(),
                                                                                    data.shape[0],
                                                                                    subcategory
                                                                                ))
                        categories_stats["category"][category]["subcategories"][subcategory]["totalclaims"] = int(subcat_mask.sum())
                        categories_stats["category"][category]["subcategories"][subcategory]["totalitems"]  = int(data.loc[subcat_mask, "item_quantity"].sum())
                    else:
                        log_and_warn("No claims belong to {} subcategory".format(subcategory))

            data = pd.DataFrame.from_dict(categories_stats)

            self.save_data_to_json(data, filename, force_ascii=False, indent=4)

        else:
            log_and_warn("File {} already exists: delete it to recreate a new one".format(filename.format(extension="json")))

    def calculate_claim_reports(self, claims, filename, report_type="yearly", data=None):

        logging.info(filename)
        filename = filename.format(type=report_type)

        if not os.path.exists(filename.format(extension="json")):

            column = None
            sheet_name = None

            data = claims.copy()

            if report_type == "yearly":
                try:
                    data['year'] = pd.DatetimeIndex(pd.to_datetime(data['ls_date'], format="%Y%m")).year
                except Exception as e:
                    logging.error("Invalid datetime format")
                    raise Exception("Failed to convert datetime column due to exception: {e}".format(e))
                
                column = "year"
                sheet_name = "Yearly"
            elif report_type == "bp":
                column = "div_cd"
                sheet_name = "Business_Personal"
            else:
                log_and_warn("Unknown type of claim report")
                return False


            data = data.assign(**{"subcategory":data["subcategory"].str.split(',')})
            data = data.explode("subcategory")

            subyearly_data = data[["subcategory", "item_quantity", "item_unit_cd", column]]\
                                .groupby(["subcategory", column, "item_unit_cd"], as_index=False)\
                                .agg(median_item_quantity=('item_quantity', np.median),total_item_quantity=('item_quantity', np.sum))
            subyearly_data = subyearly_data.reset_index().round(1)

            total_yearly_data = data[[column, "item_quantity", "item_unit_cd"]]\
                                .groupby([column, "item_unit_cd"], as_index=False)\
                                .agg(total_item_quantity=('item_quantity', np.sum))
            total_yearly_data = total_yearly_data.reset_index().round(1)
                    
            with pd.ExcelWriter(filename.format(extension="xlsx"), engine='openpyxl') as writer:
                subyearly_data.to_excel(writer, sheet_name="Subcategories")
                total_yearly_data.to_excel(writer, sheet_name=sheet_name)

            self.save_data_to_json(subyearly_data, filename.format(extension="json"))

            return True
        else:
            log_and_warn("File {} already exists: delete it to recreate a new one".format(filename.format(extension="json")))
            return True
            

    @read_if_exist_decorator
    def load_zipcode_map(self, filename, states=[], data=None):

        if data is None:
            logging.error("ZIP code database not found")
            raise Exception("Zip mapping failed due to exception: ZIP code database not found")

        try:
            data = data[constants.ZIP_DB_FIELDS]
        except Exception as e:
            logging.error("Failed to load data from ZIP code database")
            raise Exception(e)

        if len(states) > 0:
            state_mask = data["state"].isin(states)

            if state_mask.sum() > 0:
                logging.info("Found {} valid ZIP codes for states {}".format(state_mask.sum(), states))
                data = data[state_mask]

        return data

    @read_if_exist_decorator
    def calculate_zip_code_mapping(self, claims, column, filename, data=None, primary_cities=[]):

        if data is None:
            data = claims.copy()

            no_zip_mask = data["zip"].isna()

            if no_zip_mask.sum() > 0:
                log_and_warn("Total {}/{} claims have no ZIP code associated".format(no_zip_mask.sum(),
                                                                                     data.shape[0]
                                                                                     ))

                data = data[~no_zip_mask]

            data = data.assign(**{column:data[column].str.split(',')})
            data = data.explode(column)
            data = data.drop_duplicates(subset=[column, "zip"])
            data["zip"] = data["zip"].astype(int)

            if len(primary_cities) > 0 and primary_cities != ["Texas"]:
                zip_mask = data["primary_city"].isin(primary_cities)
                if zip_mask.sum() > 0:
                    log_and_warn("Total {}/{} claims are located in {}".format(zip_mask.sum(),
                                                                                    data.shape[0],
                                                                                    primary_cities
                                                                                ))
                    data = data[zip_mask]
                else:
                    log_and_warn("Zero claims are located in {}".format(primary_cities))

            data = data[[column, "zip"]].groupby([column]).agg({
                "zip": aggregate_col_values_to_comma_list
            })

            #self.save_data_to_csv(data, filename)
            self.save_data_to_excel(data, filename)

    def match_zip_codes(self, data):

        no_zip_mask = data["zip"].isna()

        if no_zip_mask.sum() > 0:
            log_and_warn("Total {}/{} claims have no ZIP code associated".format(no_zip_mask.sum(),
                                                                                 data.shape[0]
                                                                                 ))
        states = []
        zipcode_data = self.load_zipcode_map(filename=config["data"]["zip_map"], states=states)

        if zipcode_data is None:
            log_and_warn("No ZIP data loaded. Returning initial data set")
            return data

        data = data.merge(zipcode_data[["zip", "primary_city", "state", "county"]], suffixes=("_claim", "_zip"),
                          how="left", on=["zip"])

        wrong_state_mask = (data["state_claim"] != data["state_zip"])

        if wrong_state_mask.sum() > 0:
            log_and_warn("Total {}/{} claims have wrong state associated".format(wrong_state_mask.sum(),
                                                                                 data.shape[0]
                                                                                 ))
        outside_texas_mask = (data["state_zip"] != "TX")

        if outside_texas_mask.sum() > 0:
            log_and_warn("Total {}/{} claims have ZIP codes outside Texas".format(outside_texas_mask.sum(),
                                                                                  data.shape[0]
                                                                                  ))

        data = data.drop(columns=[c for c in data.columns if c.endswith("_claim")])

        data = data.rename(columns={"state_zip": "state"})

        return data

    @read_if_exist_decorator
    def match_weights_db(self, weights, claims, weights_file, filename, data=None, weight_transformations=WEIGHTS_WORD_PREPROCESS, transformations=CATEGORIES_WORD_PREPROCESS):

        if weights is None:
            log_and_warn("No weights DB is found")
            return claims

        if weight_transformations:
            word_preprocessor = Preprocessor(weight_transformations)
            weights = word_preprocessor.calculate(weights)

        weight_descriptions = weights[["primary_desc_processed", "secondary_desc_processed", "material_processed", "dimensions_processed", "values_desc_processed"]].to_dict('records')

        weights[["primary_matching", "full_matching"]] = False

        # max_weight_per_primary_desc = weights.loc[weights.groupby("primary_desc")["weight_lbs"].idxmax()].reset_index(drop=True)

        if data is None:
            data = claims.copy()

            if transformations:
                word_preprocessor = Preprocessor(transformations)
                data = word_preprocessor.calculate(data)

            data[["pentatonic_id", "weights_primary_desc", "unit"]] = ""
            data[["total_descriptions_matched", "total_words_matched"]] = 0

            total_full_desc = len(weight_descriptions)

            if total_full_desc > 0:
                logging.info("Found {} compound descriptions in weights DB".format(total_full_desc))
            else:
                logging.warning("Found no valid compound descriptions in weights DB. Exiting...".format(total_full_desc))
                return data

            item_desc = data["item_description_processed"].dropna().unique().tolist()
            category_desc = data["subcategory_prev_processed"].dropna().unique().tolist()

            total_item_desc = len(item_desc)
            if total_item_desc > 0:
                logging.info("Found {} unique item descriptions in claims DB".format(total_item_desc))
            else:
                logging.error("Found no item descriptions in claims DB!".format(total_item_desc))
                raise Exception("Weights DB matching failed due to no item descriptions found in claims DB. Check your claim data")

            total_category_desc = len(category_desc)
            if total_category_desc > 0:
                logging.info("Found {} unique category descriptions in claims DB".format(total_category_desc))
            else:
                log_and_warn("Found no category descriptions in claims DB!".format(total_category_desc))

            chunk_size = 3
            
            for desc in weight_descriptions:

                matched_mask = True
                matched_item_desc = item_desc.copy()
                matched_category_desc = category_desc.copy()
                total_descriptions_matched = 0
                total_words_matched = 0

                logging.info("=================================================")

                for key in desc:

                    if desc[key]:

                        logging.info("Processing {} '{}'...".format(key, desc[key]))

                        if ";" not in desc[key]:
                            desc_split = [" ".join(desc[key].split()[i:i+chunk_size]) for i in range(0, len(desc[key].split()), chunk_size)]
                            nwords = len([phrase for phrase in desc[key].split()])

                            if len(desc[key].split()) > 3:
                                logging.info("Length of '{}' is too long for regex search. Splitting in chunks".format(desc[key]))
                        else:
                            desc_split = [desc[key]]
                            phrases = [phrase.split() for phrase in desc[key].split(";")]
                            nwords = max([len(phrase) for phrase in phrases])

                        for desc_chunk in desc_split:     

                            logging.info("Processing chunk '{}' from {} '{}'".format(desc_chunk, key, desc[key]))

                            compiled_regex_desc = re.compile(format_and_regex(desc_chunk.lower(), permutations=True, is_synonyms=True))

                            logging.info("Regex search: {}".format(compiled_regex_desc))

                            matched_item_desc = list(filter(compiled_regex_desc.match, matched_item_desc))
                            matched_category_desc = list(filter(compiled_regex_desc.match, matched_category_desc))

                            if len(matched_item_desc) == 0 and len(matched_category_desc) == 0:
                                break

                        total_descriptions_matched += 1
                        total_words_matched += nwords

                        matched_mask &= (weights[key] == desc[key])

                        log_and_warn(
                        "Total {}/{} claim descriptions are matched with {} {} from weights DB".format(len(matched_item_desc),
                                                                                                data.shape[0],
                                                                                                key,
                                                                                                desc[key]
                                                                                                    ))
                        log_and_warn(
                        "Total {}/{} claim category descriptions are matched with {} {} from weights DB".format(len(matched_category_desc),
                                                                                                data.shape[0],
                                                                                                key,
                                                                                                desc[key]
                                                                                                    ))

                        if "primary_desc" in key and (len(matched_item_desc) > 0 or len(matched_category_desc) > 0):
                            weights.loc[matched_mask, "primary_matching"] = True

                            regex_mask = (data["item_description_processed"].isin(matched_item_desc) | data["subcategory_prev_processed"].isin(matched_category_desc))
                            data = self.add_tag(data, regex_mask, "weights_primary_desc", desc[key])

                            # primary_mask = (max_weight_per_primary_desc["primary_desc"] == desc[key])
                            # weight_lbs = max_weight_per_primary_desc.loc[primary_mask, "weight_lbs"].iloc[0]

                            # replace_mask = regex_mask & (data["max_weight_lbs"] < weight_lbs)

                            # if replace_mask.sum() > 0:
                            #     logging.info("{}: replacing {}/{} values for higher {} lbs weight values ...".format(
                            #         desc[key],
                            #         replace_mask.sum(),
                            #         regex_mask.sum(),
                            #         weight_lbs
                            #     ))

                            #     for col in max_unit_columns:
                            #         unit = max_weight_per_primary_desc.loc[primary_mask, col.replace("max_","")].iloc[0]
                            #         data = self.replace_tag(data, replace_mask, col, unit)

                total_matched_items = 0
                total_matched_item_desc = len(matched_item_desc)
                total_matched_category_desc = len(matched_category_desc)

                valid_item_matching = (total_matched_item_desc > 0 and total_matched_item_desc < total_item_desc)
                valid_category_matching = (total_matched_category_desc > 0 and total_matched_category_desc < total_category_desc)

                if valid_item_matching or valid_category_matching:

                    categories_matched_mask = (data["total_descriptions_matched"] < total_descriptions_matched) | \
                                           (data["total_descriptions_matched"] == total_descriptions_matched) & (data["total_words_matched"] < total_words_matched)
                    words_matched_mask = (data["total_descriptions_matched"] == total_descriptions_matched) & (data["total_words_matched"] == total_words_matched)

                    weights.loc[matched_mask, "full_matching"] = True

                    regex_mask = (data["item_description_processed"].isin(matched_item_desc) | data["subcategory_prev_processed"].isin(matched_category_desc))
                    total_matched_items = regex_mask.sum()

                    pentatonic_id = weights.loc[matched_mask, "pentatonic_id"].iloc[0]
                    weights_unit = weights.loc[matched_mask, "unit"].iloc[0]

                    if weights_unit:
                        regex_mask &= (data["item_unit_cd"] == weights_unit)

                    if pentatonic_id:
                        if words_matched_mask.sum() > 0:
                            data = self.add_tag(data, regex_mask & words_matched_mask, "pentatonic_id", pentatonic_id)
                        
                        if categories_matched_mask.sum() > 0:
                            data = self.replace_tag(data, regex_mask & categories_matched_mask, "pentatonic_id", pentatonic_id)
                            data.loc[regex_mask & categories_matched_mask, "total_descriptions_matched"] = total_descriptions_matched
                            data.loc[regex_mask & categories_matched_mask, "total_words_matched"] = total_words_matched
                            
                        data.loc[regex_mask, "unit"] = weights_unit

                        # for col in unit_columns:
                        #     unit = weights.loc[matched_mask, col].iloc[0]
                        #     if col == "unit":
                        #         data = self.replace_tag(data, regex_mask, col, unit, logging=False)
                        #     else:
                        #         data = self.add_tag(data, regex_mask, col, unit, logging=False)

                        # weight_lbs = weights.loc[matched_mask, "weight_lbs"].iloc[0]

                        # replace_mask = regex_mask & (data["weight_lbs"] < weight_lbs)

                        # if replace_mask.sum() > 0:
                        #     logging.info("{}: replacing {}/{} values for higher {} lbs weight values ...".format(
                        #         pentatonic_id,
                        #         replace_mask.sum(),
                        #         regex_mask.sum(),
                        #         weight_lbs                                                                                               
                        #     ))
                        #     for col in unit_columns:
                        #         unit = weights.loc[matched_mask, col].iloc[0]
                        #         data = self.replace_tag(data, replace_mask, col, unit)
                    else:
                        log_and_warn("No pentatonic ID can be assigned to matched items {} and matched descriptions {}".format(matched_item_desc, matched_category_desc))


                log_and_warn(
                        "Total {}/{} claim items are matched with weights DB".format(
                            total_matched_items,
                            data.shape[0],
                            key,
                            desc[key]
                        ))

            truck_claims = "dumpster load;pickup truck load"
            truck_claims_expr = format_and_regex(truck_claims)

            truck_mask = data["subcategory_prev_processed"].astype(str).str.contains(truck_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True) | \
                        data["item_description_processed"].astype(str).str.contains(truck_claims_expr, 
                                                                        flags=re.IGNORECASE,
                                                                        regex=True)   

            id_matched_mask = (data["pentatonic_id"].notna() & (data["pentatonic_id"] != ""))
            primary_matched_mask = (data["weights_primary_desc"].notna() & (data["weights_primary_desc"] != ""))
            only_primary_matched_mask = ~id_matched_mask & primary_matched_mask

            def get_phrase_max_length(expr):
                
                tags = expr.split(",")
                max_length = max([len(t.split()) for t in tags])

                new_expr = ",".join(t for t in tags if len(t.split()) == max_length)

                return new_expr

            data.loc[only_primary_matched_mask, "weights_primary_desc"] = data.loc[only_primary_matched_mask, "weights_primary_desc"]\
                                                                            .apply(lambda x: get_phrase_max_length(x))

            if (only_primary_matched_mask & ~truck_mask).sum() > 0:
                log_and_warn(
                        "Total {}/{} claims were matched by primary description but not Pentatonic ID...".format(
                                                                                (only_primary_matched_mask & ~truck_mask).sum(),
                                                                                data.shape[0]
                        ))

            if (id_matched_mask & ~truck_mask).sum() > 0:
                log_and_warn(
                        "Total {}/{} claims were matched by Pentatonic ID...".format(
                                                                                (id_matched_mask & ~truck_mask).sum(),
                                                                                data.shape[0]
                        ))

            # id_unmatched_mask = (data["pentatonic_id"].isna() | (data["pentatonic_id"] == ""))
            # primary_matched_mask = (data["weights_primary_desc"].notna() & (data["weights_primary_desc"] != ""))
            # valid_weights = (data["max_weight_lbs"].notna() & (data["max_weight_lbs"] > 0))

            # only_primary_matched_mask = id_unmatched_mask & primary_matched_mask & valid_weights

            # if only_primary_matched_mask.sum() > 0:
            #     log_and_warn(
            #             "Total {}/{} claims were matched by primary description but not Pentatonic ID. Replacing weights with non-zero max weight per primary desc ...".format(
            #                                                                                     only_primary_matched_mask.sum(),
            #                                                                                     data.shape[0]
            #             ))
            #     for col in unit_columns:
            #         data.loc[only_primary_matched_mask, col] = data.loc[only_primary_matched_mask, "max_"+col]

            drop_columns = [col for col in data.columns if col.endswith("_processed")]
            if drop_columns:
                data = data.drop(columns=drop_columns)

            self.save_data_to_csv(data, filename, index=False)
            self.save_data_to_excel(data, filename.replace(".csv", ".xlsx"), index=False)

            drop_columns = [col for col in weights.columns if col.endswith("_processed")]
            if drop_columns:
                weights = weights.drop(columns=drop_columns)

            self.save_data_to_csv(weights, weights_file, index=False)
            self.save_data_to_excel(weights, weights_file.replace(".csv", ".xlsx"), index=False)


        return data

    # def calculate_units(self, weights, data):

    #     unit_columns = ["weight_lbs", "weight_ustons", "volume_cf", "volume_cy"]

    #     data[unit_columns] = 0

    #     data["unit_matching"] = "undefined"
    #     valid_unit_mask = (data["unit"].notna() & (data["unit"] != ""))
    #     unit_matching_mask = (data["unit"] == data["item_unit_cd"])
    #     data.loc[unit_matching_mask & valid_unit_mask, "unit_matching"] = "yes"
    #     data.loc[~unit_matching_mask & valid_unit_mask, "unit_matching"] = "no"

    #     calculate_mask = data["pentatonic_id"].fillna("").str.contains(",")

    #     def calculate():



    def calculate_matched_match_weights_db(self, matched_claims, primary_desc, categories, filename):

        filename = filename.format(extension="xlsx")

        if not os.path.exists(filename):

            data = matched_claims.copy()

            data

        #     matching_stats = {}

        #     matching_stats["matched"] = {}
        #     matching_stats["unmatched"] = {}

        #     matching_stats["matched"]["categories"] = {}
        #     matching_stats["unmatched"]["categories"] = {}

        #     for desc in primary_desc:

        #         matching_stats[desc] = {}
        #         matching_stats[desc]["categories"] = {}

        #         matched_mask = data["weights_primary_desc"].astype(str).str.contains(desc, flags=re.IGNORECASE, regex=True)
        #         total_matched = matched_mask.sum()

        #         matching_stats[desc]["totalclaims"] = str(total_matched)
        #         matching_stats[desc]["totalitems"]  = str(data.loc[matched_mask, "item_quantity"].sum() if total_matched > 0 else 0)

        #         for category in categories:
        #             category_mask = data["category"].astype(str).str.contains(category, flags=re.IGNORECASE, regex=True)
        #             category_matched_mask = matched_mask & category_mask
        #             total_category_matched = category_matched_mask.sum()
        #             log_and_warn(
        #             "Total {}/{} claims in category {} are matched with primary description {} from weights DB".format(total_category_matched,
        #                                                                                                                 data[category_mask].shape[0],
        #                                                                                                                 category,
        #                                                                                                                 desc
        #                                                                                                             ))
        #             matching_stats[desc]["categories"][category] = {
        #                 "totalclaims" : str(total_category_matched), 
        #                 "totalitems"  : str(data.loc[category_matched_mask, "item_quantity"].sum() if total_category_matched else 0)
        #             }

        #         log_and_warn(
        #                 "Total {}/{} claims are matched with primary description {} from weights DB".format(total_matched,
        #                                                                                 data.shape[0],
        #                                                                                 desc
        #                                                                             ))

        #     unmatched_mask = (data["weights_primary_desc"] == "") | data["weights_primary_desc"].isna()

        #     matching_stats["unmatched"]["totalclaims"] = str(unmatched_mask.sum())
        #     matching_stats["matched"]["totalclaims"] = str((~unmatched_mask).sum())

        #     log_and_warn(
        #             "Total {}/{} claims are matched with primary descriptions from weights DB".format((~unmatched_mask).sum(),
        #                                                                                 data.shape[0],
        #                                                                                 desc
        #                                                                             ))

        #     for category in categories:
        #         category_mask = data["category"].astype(str).str.contains(category, flags=re.IGNORECASE, regex=True)
        #         unmatched_category_mask = category_mask & unmatched_mask
        #         matched_category_mask   = category_mask & (~unmatched_mask)

        #         matching_stats["unmatched"]["categories"][category] = str(unmatched_category_mask.sum())
        #         matching_stats["matched"]["categories"][category]   = str(matched_category_mask.sum())

        #     matching_stats_df = pd.DataFrame.from_dict(matching_stats)

        #     self.save_data_to_json(matching_stats_df, filename, force_ascii=False, indent=4)
