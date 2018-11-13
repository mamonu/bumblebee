#!/usr/bin/env python3
# -*- coding: utf-8 -*-


###########################################################################################
#### Custom Transformers to apply chain of function featurizers in a sklearn pipeline #####
###########################################################################################

# References:
# https://www.slideshare.net/PyData/julie-michelman-pandas-pipelines-and-custom-transformers
# http://fastml.com/converting-categorical-data-into-numbers-with-pandas-and-scikit-learn/

# Custom Transformers
# NOTE: BaseEstimator is included to inherit get_params() which is needed for Grid Search


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
import itertools
import numpy as np
import pandas as pd


class TextPipelineArrayFeaturizer(BaseEstimator, TransformerMixin):
    """
    A function Transformer that takes a list of (maximum 10) functions, calls each function with 
    our text (X as list of strings), and returns the results of all functions as a feature vector as np.array

    INPUT: Takes a list of maximum 10 functions, calls each function with our text (X as list of strings)
    OUTPUT: np.array
    
    Modified from https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    to make it compataible with BaseEstimator and GridSearch.
    
    """

    def __init__(self, featurizer1 = None, featurizer2 = None, featurizer3 = None, 
                 featurizer4 = None, featurizer5 = None, featurizer6 = None, featurizer7 = None, 
                 featurizer8 = None, featurizer9 = None, featurizer10 = None):
        self.featurizer1 = featurizer1 
        self.featurizer2 = featurizer2
        self.featurizer3 = featurizer3 
        self.featurizer4 = featurizer4
        self.featurizer5 = featurizer5
        self.featurizer6 = featurizer6 
        self.featurizer7 = featurizer7
        self.featurizer8 = featurizer8
        self.featurizer9 = featurizer9
        self.featurizer10 = featurizer10
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Given a list of original data, return an array of list of feature vectors."""
        featurizers = [self.featurizer1, self.featurizer2, self.featurizer3, self.featurizer4, self.featurizer5,
                      self.featurizer6, self.featurizer7, self.featurizer8, self.featurizer9, self.featurizer10]
        fvs = []
        for datum in X:
            fv = [f(datum) for f in featurizers if f is not None]
            fvs.append(fv)
        return np.array(fvs).astype(float)


class TextPipelineListFeaturizer(BaseEstimator, TransformerMixin):
    """
    A function Transformer that takes a list of (maximum 10) functions, calls each function with 
    our list of lists (X) of texts, and returns the results of all functions as a feature vector as np.array

    INPUT: List of maximum 10 functions, calls each function with our text (X as list of lists of strings)
    OUTPUT: np.array
    
    Modified from https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    and made compataible with BaseEstimator and GridSearch.
    """

    def __init__(self, featurizer1 = None, featurizer2 = None, featurizer3 = None, 
                 featurizer4 = None, featurizer5 = None, featurizer6 = None, featurizer7 = None, 
                 featurizer8 = None, featurizer9 = None, featurizer10 = None):
        self.featurizer1 = featurizer1 
        self.featurizer2 = featurizer2
        self.featurizer3 = featurizer3 
        self.featurizer4 = featurizer4
        self.featurizer5 = featurizer5
        self.featurizer6 = featurizer6 
        self.featurizer7 = featurizer7
        self.featurizer8 = featurizer8
        self.featurizer9 = featurizer9
        self.featurizer10 = featurizer10

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Given a list of lists of original data, return a list of feature vectors."""
        featurizers = [self.featurizer1, self.featurizer2, self.featurizer3, self.featurizer4, self.featurizer5,
                      self.featurizer6, self.featurizer7, self.featurizer8, self.featurizer9, self.featurizer10]
        fvs = []
        for datum in X:
            [fv] = [f(datum) for f in featurizers if f is not None]
            fvs.append(fv)
        return np.array(fvs).astype(float)


class ColumnSelector(BaseEstimator, TransformerMixin):
    """
    Class for building sklearn Pipeline step. 
    This class selects a column from a pandas data frame.
    """

    # initialise
    def __init__(self, columns):
        self.columns = columns  # e.g. pass in a column name to extract

    def fit(self, df, y=None):
        return self  # does nothing

    def transform(self, df):  # df: dataset to pass to the transformer

        df_cols = df[self.columns]  # equivelent to df['columns']
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
        self  # could also use 'pass'

    #
    def fit(self, X, y=None):
        return self

    def transform(self, X):  # X: dataset to pass to the transformer.
        Xcols_df = pd.DataFrame(X)
        Xcols_dict = Xcols_df.to_dict(orient="records")
        return Xcols_dict


class ClipTextTransformer(BaseEstimator, TransformerMixin):
    """ Selecting only first n characters of a string in a pandas dataframe column... 
        as a sklearn Transformer 
        TODO: add logic if first n chars number is larger than string itself
        
        """
    def __init__(self, n):
        self.n = n

    def fit(self, x, y=None):
        x = x[0:self.n]
        return self

    def transform(self, x):
        return x[0:self.n]




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

    def transform(self, X):  # X: dataframe to pass to the transformer.
        Xvals = X.values
        Xstrs = list(itertools.chain.from_iterable(Xvals))  # flatten nested list
        return Xstrs


class DenseTransformer(BaseEstimator, TransformerMixin):
    """
    Convert a sparse matrix (e,g,, the outcome of CountVectorizer() ) into a dense matrix, 
    required by certain classifiers in scikit-learn's Pipeline that are not compatible with sparse matrices.
    
    Ref: https://stackoverflow.com/a/28384887
    
    """

    def transform(self, X, y=None, **fit_params):
        return X.todense()

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self


class CustomCountVectorizer(CountVectorizer):
    
    """
    Customisable sklearn's CountVectorizer by overriding the build_preprocessor() and build_tokenizer()
    factory methods.
    
    Attributes:
        
        custom_preprocessor: a callable (e.g., function or chain of functions) that takes an entire document as input 
            (as a single string), and returns a transformed version of the document, still as an entire string
            Make sure: input = string; ouput = string
                        
        custom_tokeniser: a callable that takes the output from the preprocessor and splits it into tokens, 
            then returns a list of these
            Make sure: input = string; output = list of tokens
    
    ref: https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
    """
    
    def __init__(self, custom_preprocessor=None, custom_tokeniser = None, **kwargs):
        super().__init__(**kwargs)     #inherits CountVectorizer methods
        self.custom_preprocessor = custom_preprocessor if custom_preprocessor is not None else lambda x: x
        self.custom_tokeniser = custom_tokeniser if custom_tokeniser is not None else lambda x: x
    
      
    # expand the build_preprocessor method with your own customised tokenizer    
    def build_preprocessor(self):
        preprocessor = super(CustomCountVectorizer, self).build_preprocessor()
        return lambda doc : preprocessor(self.custom_preprocessor(doc))

    # expand the build_tokenizer method with your own customised tokenizer
    def build_tokenizer(self):
        tokenize = super(CustomCountVectorizer, self).build_tokenizer()
        return lambda doc: list(self.custom_tokeniser(tokenize(doc)))
