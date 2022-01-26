"""
Preprocessor and PreprocessTransformation classes
"""
import functools
import logging
import re

import numpy as np
import pandas as pd

import config as constants
from config import config

from utils.logging.helpers import log_and_warn

def remove_spaces_and_strip(func):
    """Remove extra spaces and strip values"""
    @functools.wraps(func)
    def wrapper_remove_and_strip(*args, **kwargs):
        col = func(*args, **kwargs)
        col = col.astype(str).str.replace(r"\s\s+", " ", regex=True)
        col = col.astype(str).str.strip()
        return col
    return wrapper_remove_and_strip

class Preprocessor:
    """
    Implements preprocessor to apply multiple standardized preprocess operations on a data DataFrame
    """
    def __init__(self, transformations):
        """
        :param transformations:             an iterable of `PreprocessTransformation` objects to apply
        """
        self.transformations = transformations

    def calculate(self, data):
        """
        Calculates all provided preprocessing transformations on an input DataFrame
        :param data:                        an input DataFrame
        :return:                            a DataFrame with all preprocessing transformations applied
        """
        for transformation in self.transformations:
            data[transformation.output_col] = transformation.calculate(data)
        return data


class PreprocessTransformation():
    """
    Implements preprocess transformation object that carries information about input/output columns and required
    calculation in order to standardise preprocessing framework
    """
    def __init__(self, input_cols, output_col, calculation_func, *args, **kwargs):
        """
        PreprocessorTransformation initialization
        :param input_cols:                  string or iterable of input column names for preprocessing
                                            (if iterable, it must have the same order as columns passed to
                                            `calculation_func`)
        :param output_col:                  output column name for calculated transformation
        :param calculation_func:            string or function object:
                                                - if string then correspondent transformation method from the current
                                                  class is called
                                                - otherwise it must be a calculation function that takes one or multiple
                                                  Series objects of correspondent input columns as arguments and returns
                                                  a new Series object of the same size as input
        """
        if isinstance(input_cols, str):
            self.input_cols = [input_cols]
        else:
            self.input_cols = input_cols
        self.output_col = output_col
        self.args = args
        self.kwargs = kwargs
        
        if isinstance(calculation_func, str):
            self.calculation_func = getattr(self, calculation_func)
            
    def calculate(self, df):
        """
            Performs calculations on an input DataFrame using columns defined in `self.input_cols` and calculation in
            `self.calculation_func`
            :param df:                          an input DataFrame
            :return:                            result of the transformation calculation
        """
        for input_col in self.input_cols:
            if input_col not in df.columns:
                logging.warning(
                    "{}: input column "
                    "{} wasn't found in provided DataFrame "
                    "for {} calculation".format(self.__class__.__name__, input_col, self.output_col)
                )
                return None
        return self.calculation_func(*[df[input_col] for input_col in self.input_cols], *self.args, **self.kwargs)
            
    def __repr__(self):
        """PreprocessTransformation string representation
        """
        return "PreprocessTransformation(input={}, output={})".format(self.input_cols, self.output_col)

    @staticmethod
    def identity(col):
        """Return column as is
        """
        return col

    @staticmethod
    def to_numeric(col):
        """Transforms column to numeric format
        """
        # check if column is already numeric
        if np.issubdtype(col.dtype, np.number):
            return col
        col = pd.to_numeric(col, errors="coerce")
        num_non_numeric = col.isna().sum()
        if num_non_numeric > 0:
            log_and_warn(
                "to_numeric conversion couldn't parse {} "
                "non-numeric values in `{}` column".format(num_non_numeric, col.name)
            )
        return col

    @staticmethod
    def to_datetime(col, dt_format="%Y-%m-%d"):
        """Transforms string column to datetime format
        """
        parsed = pd.to_datetime(col, format=dt_format, errors="coerce").dt.normalize()
        num_not_parsed = pd.isnull(parsed).sum() - pd.isnull(col).sum()
        if num_not_parsed > 0:
            log_and_warn(
                "to_datetime conversion couldn't parse {} "
                "dates in `{}` column, replaced with NaT".format(num_not_parsed, col.name)
            )
        return parsed

    @staticmethod
    def to_date(col, dt_format="%Y-%m-%d"):
        """Transforms string column to date format
        """
        parsed = pd.to_datetime(col, format=dt_format, errors="coerce").dt.date
        num_not_parsed = pd.isnull(parsed).sum() - pd.isnull(col).sum()
        if num_not_parsed > 0:
            log_and_warn(
                "to_date conversion couldn't parse {} "
                "dates in `{}` column, replaced with NaT".format(num_not_parsed, col.name)
            )
        return parsed


    @staticmethod
    def fill_na_with_value(col, value):
        """Fills missing values with provided value
        """
        na_num = col.isna().sum()
        if na_num > 0:
            log_and_warn("{} NA values were found in column `{}` filled with `{}`".format(na_num, col.name, value))
            return col.fillna(value=value)
        else:
            return col
        
    @staticmethod
    def fill_negative_with_value(col, value):
        """
        Fills negative values with a provided value
        :param col                          an input column Series object
        :param value:                       value to use for filling negative values
        """
        col_copy = col.copy()
        negative_values_mask = col < 0
        negative_values_num = negative_values_mask.sum()
        if negative_values_num > 0:
            log_and_warn(
                "{} negative values were found in column `{}` "
                "filled with `{}`".format(negative_values_num, col.name, value)
            )
        col_copy[negative_values_mask] = value
        return col_copy

    def fill_na_and_negatives(self, col, na_fill_value=0, negative_fill_value=0):
        """
        Convenience function for common non-negative field preprocessing. Combines filling missing values, replacing
        negative values
        :param col:                         an input column Series object
        :param na_fill_value:               value to fill NaNs
        :param negative_fill_value:         value to replace negative numbers
        """
        col = self.to_numeric(col)
        col = self.fill_na_with_value(col, na_fill_value)
        col = self.fill_negative_with_value(col, negative_fill_value)
        return self.to_numeric(col)

    @staticmethod
    def strip_and_lower(col):
        """Transforms string column to lower case and strips leading and trailing spaces
        """
        return col.astype(str).str.strip().str.lower()

    @staticmethod
    def empty_string_to_na(col):
        """Transforms string column to lower case and strips leading and trailing spaces
        """
        transformed = col.mask(col == "", np.nan)
        num_transformed = transformed.isna().sum() - col.isna().sum()
        if num_transformed > 0:
            log_and_warn(
                "{} empty strings were replaced with NaNs in column `{}`".format(num_transformed, col.name)
            )
        return transformed

    @staticmethod
    def strip_values(col):
        """
        Strips leading and trailing spaces
        :param col                          an input column Series object
        """
        return col.astype(str).str.strip()

    @staticmethod
    def remove_extra_spaces(col):
        """
        Remove multiple whitespaces with a single whitespace
        :param col                          an input column Series object
        """
        return col.astype(str).str.replace(r"\s\s+", " ", regex=True)

    @staticmethod
    @remove_spaces_and_strip
    def remove_punctuation(col):
        """
        Replace with a single whitespace:
        - All non-alphabetical and non-numerical symbols except "_", "/", ";","-"
        - Multiple whitespaces
        - All parentheses
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.astype(str).str.replace(r"[^a-zA-Z0-9\_\/\;\-\s]+", " ", regex=True)
        col_copy = col_copy.astype(str).str.replace(r"[\(\[\{\}\)\]]+", " ", regex=True)
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_stop_words(col, stopwords=constants.STOP_WORDS):
        """
        Replace stopwords by whitespace
        Default patterns:  ENGLISH_STOP_WORDS + custom stop words
        :param col                          an input column Series object
        :param stopwords                    a tuple of genre to replace
        """
        logging.info("Removing stop words: {}".format(stopwords))
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if word.strip().lower() not in stopwords])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_non_alphanumeric_words(col):
        """
        Replace words containing non-alphabetical symbols by whitespace
        Default patterns:  ENGLISH_STOP_WORDS
        :param col                          an input column Series object
        :param stopwords                    a tuple of genre to replace
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if re.sub('[/-]', '', word).isalnum()])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_one_letter_words(col):
        """
        Remove single letter words
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if len(word) > 1])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_one_letter_non_alphanumeric_words(col):
        """
        Remove single letter non-alphanumeric words
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if len(word) > 1 or (len(word) == 1 and word.isalnum())])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_numeric_words(col):
        """
        Remove numeric words
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if not re.match('[\d/-]+$', word)])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_verbs(col):
        """
        Remove words
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: [word.strip() for word in row if not (word in constants.VERBS and not word in constants.NOUNS)])
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def remove_word_duplicates(col):
        """
        Remove words
        :param col                          an input column Series object
        """
        col_copy = col.copy()
        col_copy = col_copy.str.split(" ")
        col_copy = col_copy.apply(lambda row: list(dict.fromkeys([word.strip() for word in row])))
        col_copy = col_copy.str.join(" ")
        return col_copy

    @staticmethod
    @remove_spaces_and_strip
    def add_string_columns(col1, col2):
        """
        Remove words
        :param col1, col2                    an input column Series objects
        """
        col = col1.replace(np.nan, '', regex=True) + " " + col2.replace(np.nan, '', regex=True)
        return col