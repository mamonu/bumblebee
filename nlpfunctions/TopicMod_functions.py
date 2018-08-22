#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 08:50:47 2018

@author: alessia
"""



import pandas as pd
def lda_dtm2df(lda_doc_topic_matrix, num_topics):
    
    ''' 
    Converts the doc-topic probabilities matrix into a pandas dataframe: 
    rows are documents, columns are topics storing the probabilities for each doc-topic.
        
    Input parameters:
    -----------------
    - lda_doc_topic_matrix : the doc-topic prob matrix resulting from of models.LdaModel[bow_corpus]                   
    - num_topics : number of topics used in the lda model
    
    '''    
    
    lda_doc_topic_list = list(lda_doc_topic_matrix)
    
    # for each doc, create a dataframe with "topics" (0) and "topic_probs" (1) as columns, and save them in a list of dataframes
    ts = [pd.DataFrame(lda_doc_topic_list[i]) for i in range(len(lda_doc_topic_list))]
    
    # create a (n_topic x 1) base dataframe for merging, the unique column contains the list of topics 
    df_0 = pd.DataFrame(np.array(range(0, num_topics)))
    
    # merge individual doc datasets, now each column is a document's topic probabilities
    for i in range(0, len(ts)): df_0 = pd.merge(df_0, ts[i], left_on=0, right_on=0, how='outer')
    
    df_0.index= df_0[0]
    df_0.pop(0)                     #remove column 0 (with topic indexes)
    df = df_0.fillna(0)             # fill with 0.0 (probability) where topic-doc combination is NaN (as the topic got 0 probability for that doc)
    df.columns = range(len(ts))     # add doc's idx to column (each col is a doc)
    df.sort_index(inplace=True)
    
    # transpose to that documents are rows, and topics are columns
    df = df.transpose()
    df.columns = ["t" + str(col) + "_lda" for col in df.columns]

    return df


#lda_dtm2df(lda_model[bow_corpus], 5)




def lda_topic_top_words(lda_mod, n_top_words = 6):
    
    '''
    Extract the top n words for each K topic, and convert results in a dictionary:
    - keys are the K topics
    - 2 values: first value is the list of top words, second value is the list of corresponding word probabilities
    
    Input parameter:
    ----------------
    lda mod : gensim lda model
    n_top_words : number of top words per topic to be exracted
        
    '''

    # extract n top word for each topic and their probabilties, and save them as a dictionary  
    mydict = {}
    for t in range(lda_mod.num_topics):
        ws = []
        w_probs = []
        for n in range(n_top_words):
            w, w_prob = lda_mod.show_topic(t, n_top_words)[n]
            ws.append(w)
            w_probs.append(w_prob)
        mydict[t] = ws, w_probs
    
    return mydict




def topictopwords_dict2df (topic_top_words_dict, orig_dataset = None, tech = ""):
    
    '''
    Convert the dictionary containing topics' top words into a pandas dataframe with K*2 columns:
    - k columns containing the top n word for each kth topic
    - k columns containing the associating top-word probabilities
    
    If a dataframe for the orig_dataset parameter is provided, then a final dataframe is returned 
    with the same row dimention as the orig_dataset so that they can be merged.
    
    Input parameter:
    ----------------
    - topic_top_words_dict : a dictionary with topic idx as keys, and list of top words and list of their probabilities as two values
    - orig_dataset : original dataset containing texts used for topic modelling, one text per row
    - tech (optional): a string specifying the topic modelling technique used to be added to column names (e.g., 'lda' or 'nmf')
    
    '''

    # Create dataset
    temp_topic_words_df = pd.DataFrame(topic_top_words_dict).T[0].reset_index().set_index('index').T.reset_index(drop=True)
    temp_topic_w_probs_df = pd.DataFrame(topic_top_words_dict).T[1].reset_index().set_index('index').T.reset_index(drop=True)

    temp_topic_words_df.columns = ["t" + str(col) + "_top_words_" + tech for col in temp_topic_words_df.columns]
    temp_topic_w_probs_df.columns = ["t" + str(col) + "_top_word_pbs_" + tech for col in temp_topic_w_probs_df.columns]  
    
    words_topics_df = temp_topic_words_df.join(temp_topic_w_probs_df) 
    
    # if requested, replicate dataset's rows to meet original dataset's size in view of merging
    if not orig_dataset is None:
        words_topics_df = pd.concat([words_topics_df]*(orig_dataset.shape[0]), ignore_index=True)

    return words_topics_df


#lda_topic_top_words(lda_mod = lda_model, n_top_words = 6)
#topictopwords_dict2df(lda_topic_top_words(lda_mod = lda_model, n_top_words = 6), orig_dataset = text_df, tech = 'lda')




from operator import itemgetter
def lda_ranked_topics2df(lda_mod, corpus):
    '''
    For each document, return ranked topics and their probabilities, and convert it to 
    a pandas dataframe:
        rows are documents, column "ranked_topics_lda" contains the documents' tuple of ranked topics, 
        column "ranked_topics_ps_lda" contains the tuple of associated topics' probabilities
    
    Input parameter:
    ----------------
    - lda_mod : gemsin lda model
    - corpus : bag-of-words corpus of documents resulting from [dictionary.doc2bow(text) for text in corpus]
    '''
    
    top_dict = {}
    for d_idx in range(len(corpus)): 
        top_dict[d_idx] = list(zip(*sorted(lda_mod[corpus[d_idx]], key = itemgetter(1), reverse=True)))
    
    doc_ordered_topics_df = pd.DataFrame(top_dict).T

    doc_ordered_topics_df.columns = ['ranked_topics_lda', 'ranked_topics_pbs_lda']

    return doc_ordered_topics_df


#lda_ranked_topics2df(lda_mod = lda_model, corpus = bow_corpus)




from functools import reduce
def merge_dfs(*dfs):
    '''
    Merge datasets on index. 
    Note: same index must refer to the same sample across all datasets. 
    '''
    dfs = list(dfs)
    return reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True), dfs)


#dummy = merge_dfs(text_df, tdf, doc_ordered_topics_df, words_topics_df)



import numpy as np

def standardise_twm_nmf(nmf_model):
    '''Standardise NMF topic-word matrix so that word probabilities for each doc sum up to 1'''
    
    return (nmf_model.components_/ np.sum(nmf_model.components_, axis=1, keepdims=True))

#standardise_twm_nmf(clf) 
#[np.sum(probs) for probs in standardise_twm_nmf(clf)]



def standardise_dtm_nmf(nmf_doc_topic_matrix):
    '''Standardise NMF doc_topic matrix so that the topic probabilities for each doc sum to one'''
    
    return (nmf_doc_topic_matrix / np.sum(nmf_doc_topic_matrix, axis=1, keepdims=True))




def nmf_topic_top_words(topic_term_matrix, vocabulary, n_top_words = 6):
    
    '''
    Extract the top n words for each K topic, and convert results in a dictionary:
    - keys are the K topics
    - 2 values: first value is the list of top words, second value is the list of corresponding word probabilities
    
    Input parameter:
    ----------------
    - topic_term_matrix: topic-term matrix extracting from NMF.components_ or its sandardised version
    - vocabulary : tuple of dictionary terms resulting from np.array(vectorizer.get_feature_names())
    - n_top_words : number of top words per topic to be exracted
        
    '''
    
    word_topic_dict = {}
    
    for t in range(len(topic_term_matrix)):
        topic = topic_term_matrix[t]
        word_idx = np.argsort(topic)[::-1][0:n_top_words]
        
        # topic words, topic_word_probs
        word_topic_dict[t] = [vocabulary[i] for i in word_idx], [topic[n] for n in word_idx]

    return word_topic_dict


#nmf_topic_top_words(wordtopic, vocabulary = vocab)
#topictopwords_dict2df((nmf_topic_top_words(wordtopic, vocabulary = vocab)), orig_dataset = text_df, tech = 'nmf')


from operator import itemgetter
def lda_ranked_topics2df(lda_mod, corpus):
    '''
    For each document, return ranked topics and their probabilities, and convert it to 
    a pandas dataframe:
        rows are documents, column "ranked_topics_lda" contains the documents' tuple of ranked topics, 
        column "ranked_topics_ps_lda" contains the tuple of associated topics' probabilities
    
    Input parameter:
    ----------------
    - lda_mod : gemsin lda model
    - corpus : bag-of-words corpus of documents resulting from [dictionary.doc2bow(text) for text in corpus]
    '''
   

def nmf_ranked_topics2df(nmf_doc_topic_matrix, num_topics):
    '''
    For each document, return ranked topics and their probabilities, and convert it to 
    a pandas dataframe:
        rows are documents, column "ranked_topics_nmf" contains the documents' tuple of ranked topics, 
        column "ranked_topics_ps_nmf" contains the tuple of associated topics' probabilities
    
    Input parameter:
    ----------------
    - nmf_doc_topic_matrix : NMF doc-topic matrix, or its standardised version
    - num_topics : number of topics
    '''
   
    top_dict = {} 
    for i in range(len(nmf_doc_topic_matrix)):
        top_topics = np.argsort(nmf_doc_topic_matrix[i,:])[::-1][0:num_topics]
        top_topics_probs = nmf_doc_topic_matrix[i,:][top_topics]
        top_dict[i] = top_topics, top_topics_probs
    
    ranked_docopic_df = pd.DataFrame(top_dict).T
    ranked_docopic_df.columns =  ['ranked_topics_nmf', 'ranked_topics_pbs_nmf']

    return ranked_docopic_df
          
#nmf_ranked_topics2df(doctopic, num_topics = 3)


def nmf_dtm2df(nmf_doc_topic_matrix):
    
    ''' 
    Converts the doc-topic probabilities matrix into a pandas dataframe: 
    rows are documents, columns are topics storing the probabilities for each doc-topic.
        
    Input parameters:
    -----------------
    - nmf_doc_topic_matrix : NMF doc-topic matrix, or its standardised version
    - num_topics : number of topics used in the lda model
    
    '''    
    
    doctopic_df = pd.DataFrame(nmf_doc_topic_matrix)
    doctopic_df.columns = [ str(col) + '_topic' + '_nmf' for col in doctopic_df.columns]
    
    return doctopic_df














