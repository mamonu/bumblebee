#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 17:36:22 2018

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

import os
#cwd = os.chdir('/Users/alessia/Documents/DataScience/NLP_Project/Code/functions')
#import basic_NLP_functions.utils
#dir(utils)







##############################################
### Function to sentence-tokenise text    ####
##############################################



def sent_tokenise(string_par) :   
    '''
    Function to sentence-tokenise a text paragraph of any sentence length.
    Return a list of string sentences.
    
    Parameters
    ----------
    string_par : name of the dataframe column or string that contains the paragraph text to be sentence-tokenised.
    OUTPUT : a lis of sring sentences
    '''
    try:
        return sent_tokenize(string_par)      
    except TypeError as e:
        return e
    except:
        return []




##############################################
### Function to word-tokenise sentences    ####
##############################################


def word_tokenise(list_of_strings) :
    
    """ 
    Function to word-tokenise sentences within a text.  
    Required input is a list of string sentences, e.g. ['I love dogs.', 'Me too!']
    Returns a list of lists of token words, e.g. [[ 'I', 'love', 'dogs', '.'],  ['Me', 'too', '!']]
    
    Parameters
    ----------
    list_of_strings : name of the dataframe column containing a list of string sentences in each cell or a list of string sentences.
    OUTPUT : a list of lists of token word. Each sentence's boundaries are preserved.
    """
    
    try:
        return [word_tokenize(sent) for sent in list_of_strings]
    
    except TypeError as e:
        return e
    
    except:
        return []



def to_lower(list_of_lists_of_tokens) :
    
    try:
        return [[token.lower() for token in sent] for sent in list_of_lists_of_tokens]

    except TypeError as e:
        return e
    
    except:
        return []




#####################################################################
### Function to calculate sentence-level VADER sentiment scores  ####
#####################################################################


def get_sentiment_score_VDR(list_of_strings, score_type = 'compound') :
    """ 
    Calculate nltk Vader sentiment analysis score (score_type: 'compound' default, 'pos', 'neg')
    for each sentence in a paragraph text. The input must be a list of string sentences.
    
    Return a list of scores (as float), one score for each sentence in the paragraph text.
    If text is empty, return NaN.
    
    Parameters
    ----------
    list_of_strings : name of the dataframe column or variable containing the text stored as a list of string sentences for which to 
    compute sentence-level sentiment score.
    
    score_type : 'compound' (default), 'pos' or 'neg'
    
    OUTPUT : a list of sentiment scores (as floats)
    
    """
    
    try:
        OUTPUT = np.nan if len(list_of_strings) == 0 else [analyser.polarity_scores(s)[score_type] for s in list_of_strings]
        return OUTPUT
    
    except TypeError as e:
        return e

    



##############################################
###    Function to break compound words   ####
##############################################


def break_compound_words(list_of_lists_of_tokens, compound_symbol = '-') :
    """
    Break words of the compound form word1<symbol>word2 into the constituting words, 
    then remove resulting empty strings. 
    
    Parameters
    ----------
    list_of_lists_of_tokens : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    
    compound-simbol : compound symbol word1<symbol>word2 to be broken, default is '-'
    
    OUTPUT : the original list of word-token lists with the specified compound words broken down in their components
    
    """
    
    OUTPUT = []
            
    for sent in list_of_lists_of_tokens :
                
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
    return OUTPUT




############################################################################################
###    Function to replace contracted negative forms of auxiliary verbs with negation   ####
############################################################################################


def fix_neg_auxiliary(list_of_lists_of_tokens) :
    """
    Replace contracted negative forms of auxiliary verbs with negation.
    
    Parameters
    ----------
    list_of_lists_of_tokens : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    
    OUPUT : the original list of word-token lists with the negative forms of auxiliary verbs replaced
    """
    
    OUTPUT = []
            
    for sent in list_of_lists_of_tokens :
                
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
                 
    return OUTPUT
        




############################# Define functions to remove specified stop-words ####################


def remove_stopwords(list_of_lists_of_tokens, stopwords_list=stopwords.words('english'), keep_neg = True, 
                        words_to_keep = list(), extra_stopwords = list()) :
    """
    Remove specified stop-words.
    
    Parameters
    ----------
    - list_of_lists_of_tokens : : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    - stopwords_list : (default) English stopwords from. nltk.corpus
    - keep_neg : whether to remove negation from list of stopwords, (default) True
    - words_to_keep : list of words not to remove from the text (default is empty)
    - extra_stopwords : list of ad-hoc stopwords to remove from text (default is empty)
    
    - OUTPUT : 
    
    """
    
    if keep_neg :       # keep negations in the text
        stopwords_list = [w for w in stopwords_list if not w in ['no', 'nor', 'not', "n't"]]
        
    if words_to_keep :
        stopwords_list = [w for w in stopwords_list if not w in [w.lower() for w in words_to_keep]]
        
    if extra_stopwords :
        stopwords_list += [w.lower() for w in extra_stopwords]
   
                 
    OUTPUT = [[w for w in sent if not w in stopwords_list] for sent in list_of_lists_of_tokens]
            
    return OUTPUT





############################# Function to part-of-speech tagging sentences #############################


def POS_tagging(list_of_lists_of_tokens) :
    
    """
    Return a list with POS-tags/words tuples for the specified text, using Penn Treebank 
    
    Parameters:
    -----------    
    - list_of_lists_of_tokens : : dataframe column or variable containing a list of word-token lists, with each sublist being a sentence in a paragraph text
    
    """
    
    return [pos_tag(sent) if list_of_lists_of_tokens else "" for sent in list_of_lists_of_tokens]
       





############################# Function to map the Peen Treebank tags to WordNet POS names ######


def get_wordnet_pos(treebank_tag):

    """
    Return Wordnet POS tags from Penn Treebank tags
    """
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith(('V', 'M')):    #add 'M' to verbs too?
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



def lemmatise(list_of_lists_of_pos_tuples) :
    
    """
    Return lemmas from word-POS tag tuples, using Wordnet POS tags.
    When no wordnet POS tag is avalable, it returns the original word.
    
    Parameters
    -----------
    POStag_col : dataframe column containig lists of (word, POS) tuples
    """
    
    # use the wordnet POS equivalent if it exists
    # else if the treebank POS does not have a wordnet POS equivalent, keep the original token    
            
    OUTPUT = [[wordnet_lemmatiser.lemmatize(wordPOS_tuple[0], pos=get_wordnet_pos(wordPOS_tuple[1])) if 
               get_wordnet_pos(wordPOS_tuple[1]) else wordPOS_tuple[0] for wordPOS_tuple in sent] for sent in list_of_lists_of_pos_tuples]

    return OUTPUT




   

########################## Function to remove punctuation ############################################

def remove_punctuation(list_of_string, item_to_keep = '') :
    
    """
    Remove punctuation from a list of strings.
    
    Parameters
    ----------
    - list_of_string : a dataframe column or variable containing the text stored as a list of string sentences
    - item_to_keep : a string of punctuation signs you want to keep in text (e.g., '!?.,:;')
    """
    
    # Update string of punctuation signs
    if len(item_to_keep) > 0 :
        
        punctuation_list = ''.join(c for c in string.punctuation if c not in item_to_keep)
        
    else :
        
        punctuation_list = string.punctuation
        
    # Remove punctuation from each sentence
    transtable = str.maketrans('', '', punctuation_list)
    
    return [sent.translate(transtable) for sent in list_of_string]






#################### Function to calculate subjectivity score using TextBlob ###########################################

def get_subjectivity(list_of_string):
    
    """
    Return a subjectivity score for each sentence in the input text.
    
    Parameter
    ---------
    - list_of_string : a dataframe column or variable containing the text stored as a list of string sentences for which to compute subjectivity score.
    - OUTPUT : a list of subjectivity scores for each row (one score per each sentence in the cell)
    """
        
    return [np.nan] if len(list_of_string) == 0 else [TextBlob(s).sentiment.subjectivity for s in list_of_string]




#################### Function to classify sentences based on subjectivity score ######################################

def classify_subjectivity(list_of_scores, threshold = 0.5):
    
    """
    Return a binary score (1 = subjective, 0 = objective) for each sentence in the input text 
    based on the sentence's subjectivity score. Scores > threshold are classified as subjecive.
    
    Parameter
    ---------
    - list_of_scores : a dataframe column containing a list of subjectivity scores in each row or a variable of such a lis of scores
    - threshold : the cut off value above which a sentence is classified as subjective between [0.0, 1.0] - default is 0.5
    - OUTPUT : a dataframe column consisting of a list of 1's/0's on each row
    """
    
    return [np.nan] if all(np.isnan(list_of_scores)) else [1 if s > threshold else 0 for s in list_of_scores]
        




####### Function to only keep subjective senentences in a text #############

def remove_objective_sents(list_of_strings, threshold = 0.5):
    
    """
    Return a list of lists containing only sentences with a subjective score, where 
    subjectivity is defined as > threshold.
    
    Parameter
    ---------
    - list_of_string : a dataframe column or variable containing the text stored as a list of string sentences for which to compute subjectivity score.
    - threshold : the cut off value above which a sentence is classified as subjective between [0.0, 1.0]
                default is 0.5
    - OUTPUT : a dataframe column consisting of a list of string setences in each row
    """
    
    newListOfSents = []
    
    for s in list_of_strings:
        
        if len(s) == 0 :
            newListOfSents.append(list())
            
        else :
            
            newListOfSents = [s for s in list_of_strings if TextBlob(s).sentiment.subjectivity > threshold] 
        
    return newListOfSents




####### Function to normalised scores in the 0-1 range #############

def rescale_to_01(value, min_v, max_v):
    
    """
    Returns the corresponding value in the range 0-1.
    If the original data only contains -1's and 1's, then these are return as 0's and 1's respectively.
    
    Parameters
    ----------
    value : name of the dataframe column containing the values to be scaled (one score per row)
    min_v : minimum value in the data
    max_v : maximum value in the data
    """
    return (value - min_v)/(max_v - min_v)



#########

def get_sentiment_score_TB(INPUT) :
    """ 
    Calculate sentiment analysis score 
    for each sentence in each cell (text/answer) in the specified dataframe column.
    
    Return a list of scores, one score for each sentence in the column cell.
    If text is empty, return NaN.
    
    Parameters
    ----------
    - list_of_string : a dataframe column or variable containing the text stored as a list of string sentences for which to compute sentiment score (at sentence level).
    - OUTPUT : a list of sentiment polarity score from -1 (negative) to 1 (positive), one for each sentence making up the text
    
    """
    
    OUTPUT = np.nan if len(INPUT) == 0 else [TextBlob(s).sentiment.polarity for s in INPUT]
        
    return OUTPUT



######## Function to retain only sentiment polarity scores that meet stricter threshold ######
    
def get_sentiment_stricter_threshold(list_of_scores, polarity_threshold = 0.2):
    
    """
    Return a list of lists containing only sentiment polarity scores that meet the 
    polarity threshold:

    scores > 1 * threshold*1 (for positive scores)
    scores < -1 * threshold (for negative scores)
    -1 * threshold <= score <= 1 * threshold are returned as NaN
    
    Parameters
    ----------
    - list_of_scores : a dataframe column consisting of a list of sentiment polarity scores in the range [-1, 1] in each row
    - polarity_threshold : the cut off value to consider a score as positive or negative
    - OUTPUT : a dataframe column consisting of a list of sentiment polarity scores tha meet the stricter threshold
    """
    
    OUTPUT = [np.nan] if all(np.isnan(list_of_scores)) else [s if ((s > 1*polarity_threshold) | (s < -1*polarity_threshold)) else np.nan for s in list_of_scores]
        
    return OUTPUT




####### Function to only keep senentences in a text whose sentiment polarity score meets stricter threshold #############

def keep_only_strict_polarity_sents(list_of_strings, polarity_threshold = 0.3):
    
    """
    Return a list of lists containing only sentences with a polarity score that meets the thresholds:
        if positive, score(sentence) > 1*polarity_threshold
        if negative, score(sentence) < -1*polarity_threshold
        
    Sentences' compound polarity score is calculated using nltk Vader.
    
    Parameter
    ---------
    - list_of_strings : a dataframe column consisting of a list of strings in each row, where each string is a sentence in the text.
    
    - polarity_threshold : the stricter threshold that decides whether a sentence has a positive or negative polarity, default is 0.3
    
    - OUTPUT : a list of string sentences for each row
    """
    
    newListOfSents = []
    
    for s in list_of_strings:
        
        if len(s) == 0 :
            newListOfSents.append(list())
            
        else :
            newListOfSents = [s for s in list_of_strings if (analyser.polarity_scores(s)['compound'] > 1*polarity_threshold) | (analyser.polarity_scores(s)['compound'] < -1*polarity_threshold)] 
         
    return newListOfSents





######## Function to count occurrences of specified POS as count or proportion (default) #######

import itertools
from string import punctuation

def count_pos(list_of_lists_of_pos_tuples, pos_to_cnt="", normalise = True) :
    """
    Return count or porportion of specified part-of-speech in each text
    
    Parameters
    ----------
    - list_of_lists_of_pos_tuples : dataframe column whose cells contain lists of NLTK-produced POS, 
                where each list is one sentence in a paragraph text
    - pos : part-of-speech to count, specified by their initial: 'J' for adjs, 'R' for adverbs, 'N' for nouns, 'V' for verbs
    - normalise : whether to return normalised counts (i.e., proportion), default is True
    - OUTPUT : list of integers, each being the count/proportion of pos in each paragraph text
    """
    
    # flatten list of lists in each cell, so that we have one list of tuples for each text/cell
    text_list = list(itertools.chain.from_iterable(list_of_lists_of_pos_tuples))
    
    try:
    
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
        return OUTPUT
    
    except:
        return np.nan
        




####### Function to count occurrences of specified "meaningful" punctuation symbols #####

def count_punkt(list_of_lists_of_tokens, punkt_list=[]) :
    """
    Return count of "meaningful" punctuation symbols in each text 
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column whose cells contain lists of words/tokens (one list for each sentence making up the cell text)
    - punkt_list : list of punctuation symbols to count (e.g., ["!", "?", "..."])
    - OUTPUT : pandas Series of integer, each being the count of punctuation in each text cell
   """
    
    OUTPUT = len([tok for sent in list_of_lists_of_tokens for tok in sent if tok in punkt_list])
            
    return OUTPUT






####### Function to count word (exlcuding punctuation) ######################
    
    
def count_words(list_of_lists_of_tokens, exclude_punkt = True) :
    """
    Return count of words in each text
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column whose cells contain lists of word-tokenised sentences
    - exclude_punkt : whether to exlcude punctuation symbols from the count, default is True
    - OUTPUT : pandas Series of integer, each being the count of words in each text cell
    """
    
    # flatten list of lists in each cell, so that we have one list of tuples of each text/cell
    token_list = list(flattenIrregularListOfLists(list_of_lists_of_tokens))
    
    # count number of words (incl. punkt)
    n_words = len(token_list)
    
    if not exclude_punkt :       
        
        OUTPUT = n_words
  
    else : 
        
        punkt = list(punctuation)

        punkt.extend(("''", '""', "``"))
        
        OUTPUT = len([w for w in token_list if not w in punkt])
            
    return OUTPUT





#Append _NEG suffix to words that appear in the scope between a negation and a punctuation mark.

def mark_neg(list_of_lists_of_tokens, double_neg_flip=False) :
    """
    Return count of words in each text
    
    Parameters
    ----------
    - list_of_lists_of_tokens : dataframe column whose cells contain lists of word-tokenised sentences
    - OUTPUT : 
    """
       
    return [mark_negation(sent) for sent in list_of_lists_of_tokens]




def is_part_string(text, target_string):
    """
    Returns True if targe_string is contained in text.
    Example: is_part_string('Hi @ONS', target_string='@') ==> True
    
    Input Parameters:
    -----------------
    text : a string of text
    target_string : a string of characters
    """

    return (target_string in text) if target_string else False
