
"""
Created on Fri Apr  6 22:45:21 2018

@author: alessia
"""


############################ Function to sentence-tokenise answers ############################

from nltk.tokenize import sent_tokenize
import pandas as pd

def sent_tokenise_answer_df(INPUT) :
    
    """ 
    Function to sentence-tokenise text. 
    Return a list of lists with each sublist containing an answer's sentences as strings.
    
    Parameters
    ----------
    INPUT : name of the dataframe column containing the list of sentences to be word-tokenised
    """
    
    # if no answer was provided -> return empty string list, else sent-tokenize answer
    OUTPUT = sent_tokenize(INPUT) if INPUT else list("")
            
    return pd.Series(dict(sent_tok_text = OUTPUT))


################################################################################################



############################# Function to word-tokenise sentences #############################

from nltk.tokenize import word_tokenize

def word_tokenise_answer_df(INPUT) :
    
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



################################################################################################



############################# Define function to calculate polarity score #############################

from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


def get_sentiment_score(answer_col, score_type = 'compound') :
    """ 
    Calculate sentiment analysis score (score_type: 'compound' default, 'pos', 'neg')
    for each sentence in each cell (answer) of the specified dataframe column.
    
    Return a list of scores, one score for each sentence in the column cell.
    If no answer was provided, return NA
    """
    
    sentiment_bag = [[analyser.polarity_scores(s)[score_type] for s in answer] for 
                      answer in answer_col]
        
    return pd.Series(sentiment_bag)

    


# In[4]:


# Define function to ....

import string

def break_words(answer_col, compound_symbol = '-') :
    """
    Break words of the compound form word1<symbol>word2 into the constituting words, 
    then remove empty strings. 
    
    Parameters
    ----------
    answer_col : a list of lists containing word-tokenised setences
    compound-simbol : compound symbol word1<symbol>word2 to be brokwn as string, default is '-'
    
    """
    
    # empty list collector
    tokens_bag = []
    
    for answer in answer_col :   
        
        # no answer was provided, return empty string
        if not answer : 
            tokens_bag.append("")
            
        # an answer was provided       
        else :
            
            # empty collector to keep each sentence within an answer as separate list
            words_in_s = []
            
            for sent in answer :
                
                # empty collector for words within one sentence
                words = []
                
                # 1. break words of the form word1<symbol>word2 into constituting words
            
                for w in sent :
                
                    if compound_symbol in w :
                    
                        words.extend(w.split(compound_symbol))
                    
                    else :
                    
                        words.append(w)
                    
                    # 2. Remove empty strings
                    words = list(filter(None, words))
                    
                words_in_s.append(words)

            tokens_bag.append(words_in_s)
    
    return pd.Series(tokens_bag)


# In[5]:


# Define functions to replace contracted negative forms of auxiliary verbs with negation, remove specified stop-words, 

import string
from nltk.corpus import stopwords


def fix_neg_aux(answer_col) :
    """
    Replace contracted negative forms of auxiliary verbs with negation (if True).
    
    Parameters:
    - answer_col = dataframe column whose cells contain answer texts
    """
    
    # empty list collector for all answers
    tokens_bag = []
             
    for answer in answer_col :   
        
        
        if not answer :             # no answer was provided, return empty string
            tokens_bag.append("")
            
              
        else :                      # an answer was provided 
            
            sep_sents = []
            
            for sent in answer :
                
                new_sent = []
                
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
                        
                # collect each sentence as a (separate) list of words
                sep_sents.append(new_sent)
                
            tokens_bag.append(sep_sents)
            
    return pd.Series(tokens_bag)
           


# In[6]:


# Define functions to replace contracted negative forms of auxiliary verbs with negation, remove specified stop-words, 

import string
from nltk.corpus import stopwords


def remove_stopwords(answer_col, stopwords_list=stopwords.words('english'), keep_neg = True) :
    """
    Remove specified stop-words.
    
    Parameters
    ----------
    - answer_col : dataframe column whose cells contain answer texts
    - keep_neg : whether to remove negation from list of stopwords, (default) True
    - stopwords_list : (default) English stopwords from. nltk.corpus
    """
    
    # empty list collector for all answers
    tokens_bag = []
    
    
    if keep_neg :       # keep negations in
        
        stopwords_list = [w for w in stopwords_list if not w in ['no', 'nor', 'not', 'only', 
                                                                 'up', 'down', 'further', 
                                                                 'too', 'against']]
             
            
    for answer in answer_col :   
        
        
        if not answer :             # no answer was provided, return empty string
            tokens_bag.append("")
            
              
        else :                      # an answer was provided 
            
            sep_sents = []
            
            for sent in answer :
                
                # filter out stop words from each answer
                new_sent = [w for w in sent if not w in stopwords_list]
                
                # collect each sentence as a (separate) list of words
                sep_sents.append(new_sent)
                
            tokens_bag.append(sep_sents)
            
    return pd.Series(tokens_bag)
           


# In[7]:


# Function to part-of-speech tagging sentences

from nltk import pos_tag


def POS_tagging(answer_col) :
    
    """
    Return a list with POS-tags/words tuples for the specified data column.
    
    Parameters:
    - answer_col = dataframe columns containing answer texts, as lists (answers) 
        of lists (sentences) of tokenised words
    
    """
    
    # empty list collector
    tokens_bag = []
    
    for answer in answer_col :   
        
        # no answer was provided, return empty string
        if not answer : 
            tokens_bag.append("")
            
        # an answer was provided       
        else :
            
            # empty collector for individual sentences within an asnwer
            sep_sents = []
            
            for sent in answer :
                
                # calculate Part-Of-Speech
                pos_answer = pos_tag(sent)
                
                sep_sents.append(pos_answer)
                
            
            tokens_bag.append(sep_sents)
            
    return pd.Series(tokens_bag)
                
            


# In[8]:


# TBC : should impement something like this...
# https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python

# The following function would map the Peen Treebank tags to WordNet part of speech names:
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    """
    Return Wordnet POS tags from Penn Treebank tags
    """
    
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    elif treebank_tag.startswith('S'):
        return wordnet.ADJ_SAT
    else:
        return ''


# In[9]:


# Function 

from nltk.stem import WordNetLemmatizer
wordnet_lemmatiser = WordNetLemmatizer()


# import get_wordnet_pos ?

def lemmatise(POStag_col) :
    
    """
    Return lemmas from word-POS tag tuples, using Wordnet POS tags.
    When no wornet POS tag is avalable, return the original word.
    
    Parameters:
    - POStag_col = dataframe column containig (word, POS) tuples
    """
    
    # collector for all 
    lemma_big_bag = []
    
    
    for cell in POStag_col :
        
        #print('No. of sentences (length of cell) = ' + str(len(cell)))
        
        # an answer was not provided
        if len(cell) == 0 :
            
            lemma_big_bag.append("")
            
        # an answer was provided
        else :
            
            sent_bag = []
            
            for sent in cell :
                
                print('No. of tuples (length of sent) = ' + str(len(sent)))
        
                lemma_bag = []
                
                for wordPOS_tuple in sent :
                
                    # the treebank POS does not have a wordnet POS equivalent 
                        # -> keep original token
                    if get_wordnet_pos(wordPOS_tuple[1]) == '' :
                    
                        lemma = wordPOS_tuple[0]
                    
                    # the treebank POS does have a wordnet POS equivalent
                    else :
                    
                        lemma = wordnet_lemmatiser.lemmatize(wordPOS_tuple[0], pos=get_wordnet_pos(wordPOS_tuple[1]))
                        
                    lemma_bag.append(lemma)
                    
                sent_bag.append(lemma_bag)
                    
            lemma_big_bag.append(sent_bag)
                    
            
    return pd.Series(lemma_big_bag)



# In[10]:


# Function

def detokenise_sent(word_tokenised_col) :
    
    """
    Return a list containing a single string of text for each word-tokenised sentence.
    
    Parameters:
    - word_tokenised_col = dataframe column with word-tokenised sentences
    """
    
           
    detok_sents = [[" ".join(sent) for sent in cell] for cell in word_tokenised_col]
    
    
    return(detok_sents)


# In[11]:


def list2string(list_of_lists) :
    """
    Return a string from a list of strings.
    """
    string_sents = [" ".join(mylist) for mylist in list_of_lists]

    return pd.Series(string_sents)


# In[12]:


# Function
import string 

def remove_punctuation(text_col, item_to_keep = '') :
    
    """
    Remove punctuation from a list of strings.
    
    Parameters
    ----------
    - text_col : dataframe column with text (each column cell must be a list of sentences as strings)
    - item_to_keep : a string of punctuation signs you want to keep in text (e.g., '!?.,:;')
    """
    
    # Update string of punctuation signs
    if len(item_to_keep) > 0 :
        
        punctuation_list = ''.join(c for c in string.punctuation if c not in item_to_keep)
        
    else :
        
        punctuation_list = string.punctuation
        
    # Remove punctuation from each word
    transtable = str.maketrans('', '', punctuation_list)
    
    depunct_sent = [[sent.translate(transtable) for sent in cell] for cell in text_col]

    return pd.Series(depunct_sent)

