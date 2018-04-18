import collections
import itertools

def flattenIregullarListOfListsGenerator(listOfLists):

    """ 
    Function to flatten a list of lists that is nested /irregular  eg  [1,2,[],[[3]]],4,[5,6]]
    Returns a flattened list generator 
    
    Parameters
    ----------
    INPUT : a ( nested) list of lists 
    OUTPUT : a flattened list generator 
    
    """


    for el in listOfLists:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el



def flattenListOfListsGenerator(listOfLists):
    
    """ 
    Function to flatten a list of lists that is not nested /irregular  eg  [[1,2,3],[5,6],[7,8]]
    Returns a flattened list generator
    
    Parameters
    ----------
    INPUT : a list of lists (not nested)
    OUTPUT : a flattened list generator 
    
    """

    return itertools.chain.from_iterable(listOfLists)



def flattenListOfLists_df(INPUT) :
    
    """ 
    Function to flatten a list of lists  (not nested /irregular)  eg  [[1,2,3],[5,6],[7,8]]
    that can be applied on a pandas dataframe column
    Returns a flattened list 
    
    Parameters
    ----------
    INPUT : a dataframe column consisting of (not nested) list of lists in each row 
    OUTPUT: a dataframe column with a flattened list on each row
    
    """
    

    OUTPUT = list(flattenListOfListsGenerator(INPUT))
            
    return pd.Series(dict(flatlist = OUTPUT))





def flattenIrregularListOfLists_df(INPUT) :
    
    """ 
    Function to flatten a list of lists that is nested /irregular  eg  [1,2,[],[[3]]],4,[5,6]]
    that can be applied on a pandas dataframe column
    Returns a flattened list 
    
    Parameters
    ----------
    
    INPUT : a dataframe column consisting of (not nested) list of lists in each row 
    OUTPUT: a dataframe column with a flattened list on each row



    """
    
    OUTPUT = list(flattenIregullarListOfListsGenerator(INPUT))
            
    return pd.Series(dict(flatlist = OUTPUT))
