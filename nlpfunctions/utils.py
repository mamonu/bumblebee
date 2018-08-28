#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 13:36:32 2018

@author: alessia
"""


import pandas as pd
import numpy as np



##############################################################################
### Decorators for turning the outcome of a function into a pandas.Series ####
##############################################################################

# Ref: https://gist.github.com/Zearin/2f40b7b9cfc51132851a

# option 1: it assumes the original function's ouput is not a pd.Series

def series_output(func):
    """Decorator for turning the outcome of a function into a pandas.Series """
    def wrapper(*args, **kwargs):
        return pd.Series(dict(outcome = func(*args, **kwargs)))
    return wrapper


# option 2: it does not assumes the original function's ouput is not a pd.Series

from functools import wraps

def string_to_series_out(func):
    """Decorator for turning the outcome of a function into a pandas.Series
    It first checks whether the outcome is already a pandas.Series or not"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        outcome = func(*args, **kwargs)
        if not isinstance(outcome, pd.Series):
            return pd.Series(dict(outcome = outcome))
        return outcome
    return wrapper




###############################################################################
#### Function to combine functions into a pipeline of functions ################
###############################################################################
    
import functools

# Ref. for functools.reduce: # https://docs.python.org/3/library/functools.html#functools.reduce
    
  
def combine_2fs(f, g):
    '''
    Function to chain two functions. 
    '''
    return lambda *args, **kwargs: g(f(*args), **kwargs)


def combine_functions(*f_args):
    '''
    Function to combine an n-th number of function together.
    First to last function to apply from left to right. I.e., f, g for g(f(x))
    '''
    return functools.reduce(combine_2fs, f_args, lambda x: x)




### This one works on strings and returns a pd.Series
    
def output_series(x):
        return pd.Series(dict(outcome = x))

def combine_functions_output_series(*f_args):
     
    tuple_funcs = (output_series, ) + f_args
    return functools.reduce(combine_2fs, tuple_funcs, lambda x:x)






##########################################
#### Function to detokenise sentences ####
##########################################


def word_tokens2string_sentences(list_of_lists_of_tokens) :
    
    """
    Return a list containing a single string of text for each word-tokenised sentence.
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    """
    
    return [" ".join(sent) for sent in list_of_lists_of_tokens]



 
#####################################################################################
#### Function to transform a list of lists of strings into a list of strings ########
#####################################################################################


def list2string(list_of_strings) :
    """
    Return a string from a list of strings.
    """

    return " ".join(list_of_strings)




#####################################################
#### Function to flaten irregular lists of lists ####
#####################################################


import collections

def flattenIrregularListOfLists(l):
    """ 
    Function to flatten a list of lists that is nested /irregular  eg  [1,2,[],[[3]]],4,[5,6]]
    Returns a flattened list generator 
    
    Parameters
    ----------
    INPUT : a ( nested) list of lists 
    OUTPUT : a flattened list generator 
    
    """
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str,bytes)):
            yield from flattenIrregularListOfLists(el)
    
        else:
            yield el
 
    
    

###########################################################################################
#### Custom Transformers to apply chain of function featurizers in a sklearn pipeline #####
###########################################################################################

# References: 
# https://www.slideshare.net/PyData/julie-michelman-pandas-pipelines-and-custom-transformers
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

# Custom Transformers

# NOTE: BaseEstimator is included to inherit get_params() which is needed for Grid Search
            

from sklearn.base import BaseEstimator, TransformerMixin
    
class list2array_TextFunctionFeaturizer(BaseEstimator, TransformerMixin):
    """
    https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    Takes a list of functions, calls each function with our text (X as list of strings), and 
    returns the results of all functions as a feature vector as np.array
    """
    
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Given a list of original data, return an array of list of feature vectors."""
        fvs = []
        for datum in X:
            fv = [f(datum) for f in self.featurizers]
            fvs.append(fv)
        return np.array(fvs)



class list2list_TextFunctionFeaturizer(BaseEstimator, TransformerMixin):
    """
    Modified from:
    https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    Takes a list of functions, calls each function with our list of lists (X), 
    and returns the results of all functions as a feature vector as an np.array.
    """
    
    def __init__(self, *featurizers):
        self.featurizers = featurizers

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Given a list of lists of original data, return a list of feature vectors."""
        fvs = []
        for datum in X:
            [fv] = [f(datum) for f in self.featurizers]
            fvs.append(fv)
        return np.array(fvs)




class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step. 
    This class selects a column from a pandas data frame.
    """
    
    # initialise
    def __init__(self, columns):
        self.columns = columns          # e.g. pass in a column name to extract

    def fit(self, df, y=None): 
        return self                     # does nothing

    def transform(self, df):            #df: dataset to pass to the transformer
        
        df_cols = df[self.columns]    #equivelent to df['columns']
        return df_cols

    
    
class CatToDictTransformer(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step. 
    This class turns columns from a pandas data frame (type Series) that
    contain caegorical variables into a list of dictionaries 
    that can be inputted into DictVectorizer().
    """
    
    # initialise
    def __init__(self):
        self              #could also use 'pass'
        
    # 
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        #X: dataset to pass to the transformer.
        Xcols_df = pd.DataFrame(X)
        Xcols_dict = Xcols_df.to_dict(orient = 'records')
        return Xcols_dict


import itertools

class Series2ListOfStrings(BaseEstimator, TransformerMixin):
    
    """
    Class for building sklearn Pipeline step. 
    This class turns columns from a pandas data frame (type Series) that
    contain lists of string sentences into a list of strings 
    that can be inputted into CountVectorizer().
    """
    # initialise
    def __init__(self):
        self              
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):        #X: dataframe to pass to the transformer.
        Xvals = X.values
        Xstrs = list(itertools.chain.from_iterable(Xvals))   #flatten nested list
        return Xstrs


class DenseTransformer(TransformerMixin):

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self





### examples
    
if __name__ == "__main__":
    
    
    # string
    mystring = 'I do not care. I think. Maybe not'

    # pandas.DataFrame
    data = {'par_text': ['', 'I do not care. I think. Maybe not.', 'I ate too much.', 'I can see why'], 
            'id' : [111, 222, 333, 444]}
    
    df = pd.DataFrame(data)

    # Create some basic functions

    def sent_tok(string_par) :   #expects a string  
        '''Return a sentence-tokenized version of text as a list of string sentences'''
        try:
            return  string_par.split('.')      
        except:
            return []
    
    
    def word_tok(list_of_string_sents):   #expects a list of strings
        '''Return a word-tokenized version for each sentence in a list of string sentences'''
        try:
            return [elem.split() for elem in list_of_string_sents]
        except:
            return []

    
    # function pipeline example
    # create a function pipeline    
    f_pipeline = combine_functions(word_tok, sent_tok)
    
    print( word_tok(sent_tok(mystring)) )
    print( f_pipeline(mystring)  )  #yep
    
    print( df['par_text'].apply(sent_tok).apply(word_tok) )
    print( df['par_text'].apply(f_pipeline)  )   #yep  


    

    # decorator example
    @series_output
    def sent_tok_2(text) :
        try:
            return text.split('.')     #expect a string    
        except:
            return []
        
    
    print( sent_tok_2(mystring) )
    print( type(sent_tok_2(mystring)) )



    