
import pytest
from nlpfunctions.summarization import score_keyphrases_by_textrank
from nlpfunctions.summarization import extract_candidate_words





testtext='this will install a local (non from pypi) version of the package. (now you are able to just say import nlpfunctions anywhere and it will work. no paths.'

def test_score_keyphrases_by_textrank():
    actual=score_keyphrases_by_textrank(testtext,n_keywords=0.5)
    expected=[('non', 0.1294190138916918), ('import nlpfunctions', 0.12568900848010456), ('pypi', 0.12195900306851729)]
    assert actual[1][0]==expected[1][0]

def test_extract_candidate_words():

    assert extract_candidate_words(testtext) == ['local', 'non', 'pypi', 'version', 'package', 'able', 'import', 'nlpfunctions', 'paths']