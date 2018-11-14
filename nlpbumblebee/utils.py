#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import functools
import collections
from functools import reduce, wraps


##############################################################################
### Decorators for turning the outcome of a function into a pandas.Series ####
##############################################################################

# Ref: https://gist.github.com/Zearin/2f40b7b9cfc51132851a

# option 1: it assumes the original function's ouput is not a pd.Series

def series_output(func):
    """Decorator for turning the outcome of a function into a pandas.Series """

    def wrapper(*args, **kwargs):
        return pd.Series(dict(outcome=func(*args, **kwargs)))

    return wrapper


# option 2: it does not assumes the original function's ouput is not a pd.Series

def string_to_series_out(func):
    """Decorator for turning the outcome of a function into a pandas.Series
    It first checks whether the outcome is already a pandas.Series or not"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        outcome = func(*args, **kwargs)
        if not isinstance(outcome, pd.Series):
            return pd.Series(dict(outcome=outcome))
        return outcome

    return wrapper




###############################################################################
#### Functions to combine functions into a pipeline of functions ##############
###############################################################################


# Ref. for functools.reduce: # https://docs.python.org/3/library/functools.html#functools.reduce


def combine_2fs(f, g):
    """
    Function to chain two functions. 
    """
    return lambda *args, **kwargs: g(f(*args), **kwargs)


def combine_functions(*f_args):
    """
    Function to combine an n-th number of functions together.
    First to last function to be applied from left to right. I.e., f, g for g(f(x))
    """
    return functools.reduce(combine_2fs, f_args, lambda x: x)


###


def output_series(x):
    """
    Takes strings and returns a pd.Series
    """
    return pd.Series(dict(outcome=x))


def combine_functions_output_series(*f_args):

    tuple_funcs = (output_series,) + f_args
    return functools.reduce(combine_2fs, tuple_funcs, lambda x: x)



##########################################
#### Function to detokenise sentences ####
##########################################


def word_tokens2string_sentences(list_of_lists_of_tokens):

    """
    Return a list containing a single string of text (OUTPUT) for each word-tokenised sentence (INPUT).
    
    E.g., 
    [['I', 'think', '.'], ['Therefore', ',', 'I', 'am', '.']] => ['I think .', 'Therefore , I am .']
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    """
    ## TODO: To Discuss... CHANGE NAME. Can't understand the functionality for this function from name or comment or code  . is this a FlattenList kind of thing?

    return [" ".join(sent) for sent in list_of_lists_of_tokens]



#####################################################################################
#### Function to transform a list of lists of strings into a list of strings ########
#####################################################################################


def list2string(list_of_strings):
    """
    Return a string (OUTPUT) from a list of strings (INPUT).
    
    E.g., 
    ["I think,", "Therefore, I am."] => "I think. Therefore, I am"
    """

    return " ".join(list_of_strings)



#####################################################
#### Function to flatten irregular lists of lists ###
#####################################################


def flattenIrregularListOfLists(l):
    """ 
    Function to flatten a list of lists that is nested /irregular  E.g., [1,2,[],[[3]]],4,[5,6]]
    Returns a flattened list generator 
    
    Parameters
    ----------
    INPUT : a (nested) list of lists 
    OUTPUT : a flattened list generator 
    
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flattenIrregularListOfLists(el)

        else:
            yield el


def merge_dfs(*dfs):
    """
    Function to merge datasets on index. 
    Note: the same index must refer to the same sample (i.e., row) across all datasets. 
    """
    dfs = list(dfs)
    return reduce(
        lambda left, right: pd.merge(left, right, left_index=True, right_index=True),
        dfs,
    )


