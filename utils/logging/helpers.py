"""
Helper functions for logging
"""

import os
import re
import logging
import warnings

from datetime import datetime

def log_and_warn(message):
    """
    Convenience function to generate corresponding log and warning messages about the same event
    :param message:                             text message of warning
    :return:
    """
    logging.warning(message)
    warnings.warn(message)

def log_remove_older_than(log_dir, n_days=30):
    """
    Remove all log files older than `n_days` from the specified directory. Calculation is based on today date.
    Date format in log files is assumed to be `%Y-%m-%d`
    :param log_dir:                             directory path with log files
    :param n_days:                              log files that were created earlier than this number
                                                of days will be removed
    :return:
    """
    if not os.path.exists(log_dir):
        return
    files_names = os.listdir(log_dir)
    date_reg = re.compile(r"\d{4}-\d{2}-\d{2}")
    today = datetime.today().date()
    for file_name in files_names:
        date = re.findall(date_reg, file_name)
        if len(date) == 0:
            raise ValueError(
                "Incorrect log file name. Date in format `%Y-%m-%d` wasn't found in the file name: {}".format(file_name)
            )
        elif len(date) > 1:
            raise ValueError(
                "Incorrect log file name. Multiple dates in format `%Y-%m-%d` were found in "
                "the file name: {}".format(file_name)
            )
        date = datetime.strptime(date[0], "%Y-%m-%d").date()
        days = (today - date).days
        if days >= n_days:
            os.remove(os.path.join(log_dir, file_name))


def log_initialize(file_path, file_mode, log_level, log_format_str, days_keep):
    """
    Initialize logging using BasicConfig
    :param file_path:                           output log file path. Must contain placeholder for formatting with the
                                                current date
    :param file_mode:                           specifies the mode to open the file
    :param log_level:                           root logger level
    :param log_format_str:                      format string for the handler
    :param days_keep:                           log files that were created earlier than this number
                                                of days will be removed
    :return:
    """
    file_path = file_path.format(date=datetime.today().strftime("%Y-%m-%d"))
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)
    log_remove_older_than(dir_path, n_days=days_keep)
    logging.basicConfig(
        filename=file_path,
        filemode=file_mode,
        level=log_level,
        format=log_format_str
    )

