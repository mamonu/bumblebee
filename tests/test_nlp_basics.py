
from nlpfunctions.basic_NLP_functions import sent_tokenise


def test_basics_sent_tok_hasoutput():
   
    assert sent_tokenise("this is the end. my only friend. the end") is not None



def test_basics_sent_tok_correct():
   
    assert sent_tokenise("this is the end. my only friend. the end") == ['this is the end.', 'my only friend.', 'the end']


