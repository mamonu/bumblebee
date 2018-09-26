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
import itertools
import numpy as np
import pandas as pd


class TextPipelineArrayFeaturizer(BaseEstimator, TransformerMixin):
    """
    
    Takes a list of functions, calls each function with our text (X as list of strings), and 
    returns the results of all functions as a feature vector as np.array

    

    INPUT: Takes a list of functions, calls each function with our text (X as list of strings)
    OUTPUT: np.array

    Ref: https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
    
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


class TextPipelineListFeaturizer(BaseEstimator, TransformerMixin):
    """
    Takes a list of functions, calls each function with our list of lists (X), 
    and returns the results of all functions as a feature vector as an np.array.
    
    INPUT: list of functions, calls each function with our list of lists (X)
    OUTPUT: np.array

    Modified from:
    https://dreisbach.us/articles/building-scikit-learn-compatible-transformers/
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
