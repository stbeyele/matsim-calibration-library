"""Set of utility function for data loading and manipulation"""
import os


def check_create_dir(path):
    """ Check if folder exists and if not create it """

    if not os.path.exists(path):
        os.makedirs(path)


def check_if_files_exist(paths):
    """ check if file exists"""

    exist = True
    for path in paths:
        if not os.path.exists(path):
            exist = False

    return exist


def get_value(value):

    if isinstance(value, list):
        return_value = value[0]
    else:
        return_value = value

    return return_value
