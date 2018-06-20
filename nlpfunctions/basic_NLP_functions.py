
"""
Created on Fri Apr  6 22:45:21 2018

@author: alessia
"""

import pandas as pd
import numpy as np
import string

from nltk.tokenize import sent_tokenize

from nltk.tokenize import word_tokenize

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

from nltk.stem import WordNetLemmatizer
wordnet_lemmatiser = WordNetLemmatizer()

from nltk.corpus import stopwords

from nltk import pos_tag

from nltk.corpus import wordnet

from textblob import TextBlob

from nltk.sentiment.util import mark_negation





############################ Function to sentence-tokenise answers ############################


def sent_tokenise_df(INPUT) :
    
    """ 
    Function to sentence-tokenise text. 
    Return a list of lists with each sublist containing an answer's sentences as strings.
    
    Parameters
    ----------
    INPUT : name of the dataframe column containing the list of sentences to be word-tokenised
    """
    
    # if no answer was provided -> return empty string list, else sent-tokenize answer
    OUTPUT = sent_tokenize(INPUT) if (INPUT and isinstance(INPUT, str)) else list()
            
    return pd.Series(dict(sent_tok_text = OUTPUT))




############################# Function to word-tokenise sentences #############################

def word_tokenise_df(INPUT) :
    
    """ 
    Function to word-tokenise sentences within a text. 
    Return a list of lists of lower-case words as strings. 
    Required input, a list of lists, with each sublist containing sentences as strings.
    
    Parameters
    ----------
    INPUT : name of the dataframe column, a list of lists containing the sentences to be word-tokenised
    """
    
    # If an answer was provided: 1. word-tokenise the answer 2. convert to lower case
    OUTPUT = [[w.lower() for w in word_tokenize(sent)] for sent in INPUT]
          
    return pd.Series(dict(word_tok_sents = OUTPUT))





############################# Define function to calculate polarity score #############################


def get_sentiment_score_df(INPUT, score_type = 'compound') :
    """ 
    Calculate sentiment analysis score (score_type: 'compound' default, 'pos', 'neg')
    for each sentence in each cell (text/answer) in the specified dataframe column.
    
    Return a list of scores, one score for each sentence in the column cell.
    If text is empty, return NaN.
    
    Parameters
    ----------
    INPUT : name of the dataframe column containing the text for which to 
    compute sentiment score (at senence level).
    
    score_type : 'compound' (default), 'pos' or 'neg'
    
    """
    
    OUTPUT = np.nan if len(INPUT) == 0 else [analyser.polarity_scores(s)[score_type] for s in INPUT]
        
    return pd.Series(dict(SA_scores_sents = OUTPUT))

    



############################# Define function to break compound-words #############################


def break_words_df(INPUT, compound_symbol = '-') :
    """
    Break words of the compound form word1<symbol>word2 into the constituting words, 
    then remove empty strings. 
    
    Parameters
    ----------
    INPUT : dataframe column as a list of sublists with each sublist containing a word-tokenised setence
    
    compound-simbol : compound symbol word1<symbol>word2 to be broken, default is '-'
    
    """
    
    OUTPUT = []
            
    for sent in INPUT :
                
        # empty collector for words within each sentence
        words = []
                
        for w in sent :
            
            # 1. break words of the form word1<symbol>word2 into constituting words
            if compound_symbol in w :
                    
                words.extend(w.split(compound_symbol))
                    
            else :
                    
                words.append(w)
                    
            # 2. Remove empty strings
            words = list(filter(None, words))
                    
        OUTPUT.append(words)

            
    return pd.Series(dict(word_tok_text = OUTPUT))





##### Function to replace contracted negative forms of auxiliary verbs with negation, remove specified stop-words #########


def fix_neg_aux_df(INPUT) :
    """
    Replace contracted negative forms of auxiliary verbs with negation.
    
    Parameters
    ----------
    INPUT : dataframe column whose cells contain text
    """
    
    OUTPUT = []
            
    for sent in INPUT :
                
        new_sent = []   #collector to keep each sentence as a separate list
                
        for w in sent :
                        
            if w in ["don't", "didn", "didn't", "doesn", "doesn't", 'hadn', "n't",
                             "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', 
                             "isn't", 'mightn', "mightn't", 'mustn', "mustn't", 
                             'needn', "needn't", "shan't", 'shouldn', "shouldn't", 
                             'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
                             'wouldn', "wouldn't", 'aren', "aren't", 'couldn', "couldn't"] :          
                w = 'not'
                        
            else :
                        
                w = w
                        
            new_sent.append(w)
                         
        OUTPUT.append(new_sent)
                 
    return pd.Series(dict(word_tok_text = OUTPUT))
        




############################# Define functions to remove specified stop-words ####################


def remove_stopwords_df(INPUT, stopwords_list=stopwords.words('english'), keep_neg = True) :
    """
    Remove specified stop-words.
    
    Parameters
    ----------
    - INPUT : dataframe column whose cells contain text
    - keep_neg : whether to remove negation from list of stopwords, (default) True
    - stopwords_list : (default) English stopwords from. nltk.corpus
    """
    
    if keep_neg :       # keep negations in the text
        
        stopwords_list = [w for w in stopwords_list if not w in ['no', 'nor', 'not', 'only', 
                                                                 'up', 'down', 'further', 
                                                                 'too', 'against']]
                 
    OUTPUT = [[w for w in sent if not w in stopwords_list] for sent in INPUT]
            
            
    return pd.Series(dict(word_tok_nostopw_text = OUTPUT))





############################# Function to part-of-speech tagging sentences #############################


def POS_tagging_df(INPUT) :
    
    """
    Return a list with POS-tags/words tuples for the specified data column.
    
    Parameters:
    -----------    
    INPUT : dataframe columns containing answer texts, as lists (answers) 
    of lists (sentences) of tokenised words
    
    """
    
    OUTPUT = [pos_tag(sent) if INPUT else "" for sent in INPUT]

    return pd.Series(dict(pos_tags = OUTPUT))
       





############################# Function to map the Peen Treebank tags to WordNet POS names ######


def get_wordnet_pos(treebank_tag):

    """
    Return Wordnet POS tags from Penn Treebank tags
    """
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):    #add 'M' to verbs too?
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('S'):
        return wordnet.ADJ_SAT
    else:
        return ''






############################# Define function to lemmatise words ################################


# import get_wordnet_pos ?

def lemmatise_df(INPUT) :
    
    """
    Return lemmas from word-POS tag tuples, using Wordnet POS tags.
    When no wornet POS tag is avalable, return the original word.
    
    Parameters
    -----------
    POStag_col : dataframe column containig lists of (word, POS) tuples
    """
    
    # use the wordnet POS equivalent if it exists
    # the treebank POS does not have a wordnet POS equivalent -> keep original token    
            
    OUTPUT = [[wordnet_lemmatiser.lemmatize(wordPOS_tuple[0], pos=get_wordnet_pos(wordPOS_tuple[1])) if 
               get_wordnet_pos(wordPOS_tuple[1]) else wordPOS_tuple[0] for wordPOS_tuple in sent] for sent in INPUT]

    return pd.Series(dict(lemmas_sent = OUTPUT))




########################## Function to detokenise sentences ####################################

def word_detokenise_sent_df(INPUT) :
    
    """
    Return a list containing a single string of text for each word-tokenised sentence.
    
    Parameters
    ----------
    INPUT : a dataframe column consisting of a list of lists in each row, where each sublist is a word-tokenised sentence
    """
    
    OUTPUT = [" ".join(sent) for sent in INPUT]
    
    return pd.Series(dict(detok_sents = OUTPUT))



 

########################## Function to transform a list of lists of strings into a list of strings ########

def list2string_df(INPUT) :
    """
    Return a string from a list of strings.
    """
    OUTPUT = [" ".join(INPUT)]

    return pd.Series(dict(list_of_strings = OUTPUT))



 

########################## Function to remove punctuation ############################################

def remove_punctuation_df(INPUT, item_to_keep = '') :
    
    """
    Remove punctuation from a list of strings.
    
    Parameters
    ----------
    - INPUT : a dataframe column consisting of a list of sentences (as strings)
    - item_to_keep : a string of punctuation signs you want to keep in text (e.g., '!?.,:;')
    """
    
    # Update string of punctuation signs
    if len(item_to_keep) > 0 :
        
        punctuation_list = ''.join(c for c in string.punctuation if c not in item_to_keep)
        
    else :
        
        punctuation_list = string.punctuation
        
    # Remove punctuation from each word
    transtable = str.maketrans('', '', punctuation_list)
    
    OUTPUT = [sent.translate(transtable) for sent in INPUT] 

    return pd.Series(dict(no_punkt_sents = OUTPUT))






#################### Function to calculate subjectivity score using TextBlob ###########################################

def get_subjectivity_df(INPUT):
    
    """
    Return a subjectivity score for each sentence in the input text.
    
    Parameter
    ---------
    INPUT : a dataframe column consisting of a list of sentences (as strings) for which to 
            compute subjectivity score.
    OUTPUT : a list of subjectivity scores for each row (one score per each sentence in the cell)
    """
    
    OUTPUT = [np.nan] if len(INPUT) == 0 else [TextBlob(s).sentiment.subjectivity for s in INPUT]
        
    return pd.Series(dict(Subj_scores_sents = OUTPUT))




#################### Function to classify sentences based on subjectivity score ######################################

def classify_subjectivity_df(INPUT, threshold = 0.5):
    
    """
    Return a binary score (1 = subjective, 0 = objective) for each sentence in the input text 
    based on the sentence's subjectivity score. Scores > threshold are classified as subjecive.
    
    Parameter
    ---------
    INPUT : a dataframe column consisting of a list of subjectivity scores in each row
    threshold : the cut off value above which a sentence is classified as subjective between [0.0, 1.0]
                default is 0.5
    OUTPUT : a dataframe column consisting of a list of 1's/0's on each row
    """
    
    OUTPUT = [np.nan] if all(np.isnan(INPUT)) else [1 if s > threshold else 0 for s in INPUT]
        
    return pd.Series(dict(subjective_sents = OUTPUT))




####### Function to only keep subjective senentences in a text #############

def remove_objective_sents_df(listOfSents, threshold = 0.5):
    
    """
    Return a list of lists containing only sentences with a subjective score, where 
    subjectivity is defined as > threshold.
    
    Parameter
    ---------
    INPUT : a dataframe column consisting of a list of strings in each row, where each string is a sentence in the text.
    threshold : the cut off value above which a sentence is classified as subjective between [0.0, 1.0]
                default is 0.5
    OUTPUT : a dataframe column consisting of a list of setences as strings for each row
    """
    
    newListOfSents = []
    
    for s in listOfSents:
        
        if len(s) == 0 :
            newListOfSents.append(list())
            
        else :
            
            newListOfSents = [s for s in listOfSents if TextBlob(s).sentiment.subjectivity > threshold] 
        
         
    return pd.Series(dict(only_subject_sents = newListOfSents))




####### Function to normalised scores in the 0-1 range #############

def rescale_to_01_df(value, min_v, max_v):
    
    """
    Returns the corresponding value in the range 0-1.
    If the original data only contains -1's and 1's, then these are return as 0's and 1's respectively.
    
    Parameters
    ----------
    value : name of the dataframe column containing the values to be scaled (one score per row)
    min_v : minimum value in the data
    max_v : maximum value in the data
    """
    
    rescaled = (value - min_v)/(max_v - min_v)
    
    return pd.Series(dict(rescaled_value = rescaled))



#########

def get_textblob_sentiment_score_df(INPUT) :
    """ 
    Calculate sentiment analysis score 
    for each sentence in each cell (text/answer) in the specified dataframe column.
    
    Return a list of scores, one score for each sentence in the column cell.
    If text is empty, return NaN.
    
    Parameters
    ----------
    INPUT : name of the dataframe column containing the text for which to 
    compute sentiment score (at sentence level).
    
    OUTPUT : the sentiment polarity score from -1 (negative) to 1 (positive)
    
    """
    
    OUTPUT = np.nan if len(INPUT) == 0 else [TextBlob(s).sentiment.polarity for s in INPUT]
        
    return pd.Series(dict(SA_scores_sents = OUTPUT))



######## Function to retain only sentiment polarity scores that meet stricter threshold ######
    
def get_sentiment_stricter_threshold_df(INPUT, polarity_threshold = 0.2):
    
    """
    Return a list of lists containing only sentiment polarity scores that meet the 
    polarity threshold:

    scores > 1 * threshold*1 (for positive scores)
    scores < -1 * threshold (for negative scores)
    -1 * threshold <= score <= 1 * threshold are returned as NaN
    
    Parameter
    ---------
    INPUT : a dataframe column consisting of a list of sentiment polarity scores in the range [-1, 1] in each row
    polarity_threshold : the cut off value to consider a score as positive or negative
    OUTPUT : a dataframe column consisting of a list of sentiment polarity scores tha meet the
            stricter threshold
    """
    
    OUTPUT = [np.nan] if all(np.isnan(INPUT)) else [s if ((s > 1*polarity_threshold) | (s < -1*polarity_threshold)) else np.nan for s in INPUT]
        
    return pd.Series(dict(stricter_sent_scores = OUTPUT))




####### Function to only keep senentences in a text whose sentiment polarity score meets stricter threshold #############

def keep_only_strict_polarity_sents_df(listOfSents, polarity_threshold = 0.3):
    
    """
    Return a list of lists containing only sentences with a polarity score that meets the thresholds:
        if positive, score(sentence) > 1*polarity_threshold
        if negative, score(sentence) < -1*polarity_threshold
        
    Sentences' compound polarity score is calculated using nltk Vader.
    
    Parameter
    ---------
    INPUT : a dataframe column consisting of a list of strings in each row, where each string is a sentence in the text.
    
    polarity_threshold : the stricter threshold that decides whether a sentence has a positive or negative polarity, default is 0.3
    
    OUTPUT : a dataframe column consisting of a list of sentences as strings for each row
    """
    
    newListOfSents = []
    
    for s in listOfSents:
        
        if len(s) == 0 :
            newListOfSents.append(list())
            
        else :
            
            newListOfSents = [s for s in listOfSents if (analyser.polarity_scores(s)['compound'] > 1*polarity_threshold) | (analyser.polarity_scores(s)['compound'] < -1*polarity_threshold)] 
        
         
    return pd.Series(dict(strict_polarity_sents = newListOfSents))





######## Function to count occurrences of POS as count or proportion (default) #######

import itertools
from string import punctuation

def count_pos_df(INPUT, pos_to_cnt="", normalise = True) :
    """
    Return count or porportion of specified part-of-speech in each text
    
    Parameters
    ----------
    - INPUT : dataframe column whose cells contain lists of NLTK-produced POS, 
                where each list is one sentence in the cell text
    - pos : part-of-speech to count, specified by their initial: 'J' for adjs, 'R' for adverbs, 'N' for nouns, 'V' for verbs
    - normalise : whether to return normalised counts (i.e., proportion), default is True
    - OUTPUT : pandas Series of integer, each being the count/proportion of pos in each text cell
    """
    
    # flatten list of lists in each cell, so that we have one list of tuples for each text/cell
    text_list = list(itertools.chain.from_iterable(INPUT))
    
    # separate words from tags
    words, tags = zip(*text_list)
    
    # count of POS
    pos_cnt = len([mypos for mypos in list(tags) if mypos.startswith(pos_to_cnt)])


    if normalise :
        
        # count number of words (incl. punkt)
        n_words = len(words)
    
        # count punctuations
        n_punkt = len([mypos for mypos in list(tags) if mypos in punctuation])

        # count of "real words"
        n_real_words = n_words - n_punkt

        # prop of POS
        pos_prop = round(pos_cnt/n_real_words, 2)
        
        OUTPUT = pos_prop
        
    else : OUTPUT = pos_cnt
            
    return pd.Series(dict(pos_count = OUTPUT))




####### Function to count occurrences of specified "meaningful" punctuation symbols #####

def count_punkt_df(INPUT, punkt_list="") :
    """
    Return count of occurrences of "meaningful" punctuation symbols in each text 
    
    Parameters
    ----------
    - INPUT : dataframe column whose cells contain a string of text
    - punkt_list : string of punctuation symbols to count (e.g., ":,!?")
    - OUTPUT : pandas Series of integer, each being the count of punctuation in each text cell
    """
                 
    OUTPUT = [len([cha for cha in str(INPUT) if cha in punkt_list])]
            
    return pd.Series(dict(count_punkt = OUTPUT))




####### Function to count word (exlcuding punctuation) ######################

import collections
def flattenIrregularListOfLists(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str,bytes)):
            yield from flattenIrregularListOfLists(el)
    
        else:
            yield el
    
    
    
def count_words_df(INPUT, exclude_punkt = True) :
    """
    Return count of words in each text
    
    Parameters
    ----------
    - INPUT : dataframe column whose cells contain lists of word-tokenised sentences
    - exclude_punkt : whether to exlcude punctuation symbols from the count, default is True
    - OUTPUT : pandas Series of integer, each being the count of words in each text cell
    """
    
    # flatten list of lists in each cell, so that we have one list of tuples of each text/cell
    token_list = list(flattenIrregularListOfLists(INPUT))
    
    # count number of words (incl. punkt)
    n_words = len(token_list)
    
    if not exclude_punkt :       
        
        OUTPUT = n_words
  
    else : 
        
        punkt = list(punctuation)

        punkt.extend(("''", '""', "``"))
        
        OUTPUT = len([w for w in token_list if not w in punkt])
            
    return pd.Series(dict(word_count = OUTPUT))





#Append _NEG suffix to words that appear in the scope between a negation and a punctuation mark.

def mark_neg_df(INPUT, double_neg_flip=False) :
    """
    Return count of words in each text
    
    Parameters
    ----------
    - INPUT : dataframe column whose cells contain lists of word-tokenised sentences
    - OUTPUT : 
    """
    
    # flatten list of lists in each cell, so that we have one list of tuples of each text/cell
    token_list = list(flattenIrregularListOfLists(INPUT))
    
    OUTPUT = [mark_negation(sent) for sent in INPUT]
       
    return pd.Series(dict(tok_word_mark_neg = OUTPUT))


