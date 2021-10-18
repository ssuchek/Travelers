import collections
import ijson
import itertools
import json
import logging
import os
import re
import requests
import shutil
from utils.logging.helpers import log_and_warn

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import config as constants
from config import config

from utils.preprocess import Preprocessor, PreprocessTransformation
from utils.preprocess import BASIC_PREPROCESS
from utils.preprocess.dataframe import aggregate_col_values_to_comma_list

from utils.helpers import format_and_regex


def read_if_exist_decorator(get_data):
    def _read_file_if_exists(*args, **kwargs):
        try:
            filename = kwargs["filename"]
            if os.path.exists(filename):
                logging.info("Loading from file {}...".format(filename))
                try:
                    data = pd.read_csv(filename)
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

        if data is None:
            data = claims.copy()

            basic_preprocessor = Preprocessor(BASIC_PREPROCESS)
            data = basic_preprocessor.calculate(data)

            stop_claims = "|".join([format_and_regex(p) for p in constants.ALL_STOP_CLAIMS])

            stop_claims_mask = data["subcategory_prev"].astype(str).str.contains(stop_claims, flags=re.IGNORECASE,
                                                                                 regex=True) | data[
                                   "item_description"].astype(str).str.contains(stop_claims, flags=re.IGNORECASE,
                                                                                regex=True)

            if stop_claims_mask.sum() > 0:
                log_and_warn(
                    "Total {}/{} claims are filtered out according to ALL_STOP_CLAIMS".format(stop_claims_mask.sum(),
                                                                                              data.shape[0]
                                                                                              ))
                data = data[~stop_claims_mask]

            data = self.match_zip_codes(data)

            self.save_data_to_csv(data, filename, index=False)

        return data

    def most_frequent_words(self, data, filename):

        claims = data.copy()

        claims["item_description"] = claims["item_description"].astype(str).str.split()

        nested_item_list = claims["item_description"].values.tolist()

        item_list = list(itertools.chain.from_iterable(nested_item_list))

        words = collections.Counter(item_list)

        total_words = len(set(words))

        logging.info("Total unique words: {}".format(total_words))

        most_frequent_words = pd.DataFrame(words.most_common(total_words),
                                           columns=['words', 'count'])

        self.save_data_to_csv(most_frequent_words, filename.format(extension="csv"), index=False)
        self.save_data_to_excel(most_frequent_words, filename.format(extension="xlsx"), index=False)

        return most_frequent_words

    def plot_most_frequent_words(self, data, filename):

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

    def add_tag(self, data, mask, tag_col, tag):
        """
        Aggregates multiple tags in the single value of type tag1,tag2,tag3,...
        :param data:           an input DataFrame
        :param mask:            an input DataFrame mask     
        :param tag_col:         name of the tag column
        :param tag:             an input tag to incorporate in tag column
        """
        tag_mask = data[tag_col].fillna("").str.contains(tag + ",") | data[tag_col].fillna("").str.contains(
            "," + tag) | (data[tag_col] == tag)
        log_and_warn("{}/{} items are previously tagged as {}: no need to retag".format((mask & tag_mask).sum(),
                                                                                        data.shape[0],
                                                                                        tag
                                                                                        ))

        log_and_warn("Tagged {}/{} items as {}".format((mask & (~tag_mask)).sum(),
                                                       data.shape[0],
                                                       tag
                                                       ))

        def add_tag_to_str(x, tag):

            if x == "":
                return tag

            if not tag + "," in x and not "," + tag in x:
                x = x + "," + tag

            return x

        if (mask & (~tag_mask)).sum() > 0:
            data.loc[mask & (~tag_mask), tag_col] = data.loc[mask & (~tag_mask), tag_col].fillna("").apply(
                lambda x: add_tag_to_str(x, tag))

        return data

    @read_if_exist_decorator
    def calculate_categories(self, claims, categories, filename, data=None):

        if data is None:
            data = claims.copy()
            data["category"] = ""
            data["subcategory"] = ""
            for category, subcategory_map in categories.items():
                logging.info("Processing category {}...".format(category))
                for subcategory, pattern in subcategory_map.items():
                    logging.info("Processing subcategory {}...".format(subcategory))
                    logging.info("Search pattern: {}".format(pattern))
                    subcategory_mask = data["subcategory_prev"].astype(str).str.contains(pattern, flags=re.IGNORECASE, regex=True) | \
                                        data["item_description"].astype(str).str.contains(pattern, flags=re.IGNORECASE,regex=True)
                    if subcategory_mask.sum() > 0:
                        log_and_warn("Total {}/{} items belong to subcategory {}".format(subcategory_mask.sum(),
                                                                                         data.shape[0],
                                                                                         subcategory
                                                                                         ))
                        data = self.add_tag(data, subcategory_mask, "category", category)
                        data = self.add_tag(data, subcategory_mask, "subcategory", subcategory)
                category_mask = (data["category"] == category)
                log_and_warn("Total {}/{} items belong to category {}".format(category_mask.sum(),
                                                                              data.shape[0],
                                                                              category
                                                                              ))
            self.save_data_to_csv(data, filename, index=False)

        return data

    def calculate_categories_stats(self, data, categories, filename, states=["TX"]):

        if not os.path.exists(filename.format(extension="json")):
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

            data = pd.DataFrame.from_dict(categories_stats, orient='index')

            self.save_data_to_excel(data, filename.format(extension="xlsx"), sheet_name="Stats")
            self.save_data_to_json(data, filename.format(extension="json"), force_ascii=False, indent=4)

        else:
            log_and_warn("File {} already exists: delete it to recreate a new one".format(filename.format(extension="json")))

    def calculate_claim_reports(self, claims, filename, type="yearly", data=None):

        filename = filename.format(type=type)

        if not os.path.exists(filename.format(extension="json")):

            column = None
            sheet_name = None

            data = claims.copy()

            if type == "yearly":
                try:
                    data['year'] = pd.DatetimeIndex(pd.to_datetime(data['ls_date'], format="%Y%m")).year
                except Exception as e:
                    logging.error("Invalid datetime format")
                    raise Exception("Failed to convert datetime column due to exception: {e}".format(e))
                
                column = "year"
                sheet_name = "Yearly"
            elif type == "bp":
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

            if len(primary_cities) > 0:
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
