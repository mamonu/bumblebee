import pandas as pd
import pytest
from nlpfunctions.basicnlp import sent_tokenise
from nlpfunctions.utils import combine_functions,output_series,combine_2fs,word_tokens2string_sentences,list2string



# string
mystring = "i do not care. i think that this is bad but maybe not."



def capital_case(x):
    if not isinstance(x, str):
        raise TypeError('Please provide a string argument')
    return x.capitalize()





f_pipeline = combine_functions(capital_case,sent_tokenise)

def test_combine_functions():
    assert sent_tokenise(capital_case(mystring)) == f_pipeline(mystring)  



f_pipeline2 =  combine_2fs(capital_case, sent_tokenise)

def test_combine_2fs():
    assert sent_tokenise(capital_case(mystring)) == f_pipeline2(mystring) 



def test_output_series():
    assert  output_series(mystring).all() == (pd.Series(dict(outcome=mystring))).all()



def test_word_tokens2string_sentences():
    assert ['i do not care.', 'i think that this is bad but maybe not.']==word_tokens2string_sentences([["i", "do", "not", "care."],["i", "think", "that", "this", "is", "bad", "but", "maybe", "not."]])