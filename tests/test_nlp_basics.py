
from nlpfunctions.basicnlp import (
    sent_tokenise,
    word_tokenise,
    get_sentiment_score_VDR,
    get_sentiment_score_TB,
    get_subjectivity,
    remove_objective_sents,
    remove_stopwords,
    remove_punctuation,
    remove_objective_sents,
    break_compound_words,
    to_lower
)
from nlpfunctions.utils import *
from hypothesis import given
from hypothesis.strategies import text
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import TweetTokenizer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_raises
from nltk.stem import WordNetLemmatizer
import pytest 


def test_basics_sent_tok_hasoutput():

    assert sent_tokenise("this is the end. my only friend. the end") is not None




def test_basics_sent_tok_exception():
        with pytest.raises(TypeError):
            sent_tokenise(True,2)



def test_basics_sent_tok_correct():

    assert sent_tokenise("this is the end. my only friend. the end") == [
        "this is the end.",
        "my only friend.",
        "the end",
    ]


def test_basics_sent_tok_and_Vader():

    assert (
        np.mean(
            get_sentiment_score_VDR(
                sent_tokenise(
                    "I know that sounds funny, but to me it seemed like sketchy technology that wouldn't work well. Well, this one works great."
                )
            )
        )
        == 0.595050
    )


def test_basics_sent_tok_and_Vader_in_pipeline():

    par_sentiment_VDR_fn = combine_functions(
        sent_tokenise, get_sentiment_score_VDR, np.mean
    )
    assert (
        par_sentiment_VDR_fn(
            "I know that sounds funny, but to me it seemed like sketchy technology that wouldn't work well. Well, this one works great."
        )
        == 0.595050
    )


def test_basics_sent_tok_and_TB_in_pipeline():

    par_sentiment_TB_fn = combine_functions(
        sent_tokenise, get_sentiment_score_TB, np.mean
    )
    assert (
        par_sentiment_TB_fn(
            "I know that sounds funny, but to me it seemed like sketchy technology that wouldn't work well. Well, this one works great."
        )
        == 0.525
    )


@given(text())
def test_wordtokenise_has_output(s):
    assert word_tokenise(s) is not None



def test_wordtokenise_raises_exception():
        with pytest.raises(TypeError):
            word_tokenise(True,2)



def test_NLTK_SnowballStemmer():
    stemmer = SnowballStemmer('english')
    assert stemmer.stem("y's") == 'y'


def test_NLTK_tweet_tokenizer():
        """
        Test TweetTokenizer using words with special and accented characters.
        """
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        s9 = "@myke: Let's test these words: resumé España München français"
        tokens = tokenizer.tokenize(s9)
        expected = [':', "Let's", 'test', 'these', 'words', ':', 'resumé',
                    'España', 'München', 'français']
        assert tokens == expected


def test_remove_stopwords():

        wish = "And did they get you to trade Your heros for ghosts? Hot ashes for trees? Hot air for a cool breeze?" 
        wishtokens_sw_removed = remove_stopwords(word_tokenise(sent_tokenise(wish)))
        assert wishtokens_sw_removed == [['And', 'get', 'trade', 'Your', 'heros', 'ghosts', '?'],['Hot', 'ashes', 'trees', '?'],['Hot', 'air', 'cool', 'breeze', '?']]


def test_remove_punctuation():
    assert remove_punctuation(["And did they get you to trade Your heros for ghosts?"])==['And did they get you to trade Your heros for ghosts']


def test_to_lower():
    assert to_lower([["CRYPTANALYST"]]) == [['cryptanalyst']]


def test_break_compound_words():
    assert break_compound_words([["industry-standard"]]) == [['industry', 'standard']]


def test_no_break_non_compound_words():
    assert break_compound_words([["industry standard"]]) == [['industry standard']]


def test_get_subjectivity():
    assert [0.0, 0.8500000000000001, 0.75] == get_subjectivity(["And did they get you to trade Your heros for ghosts?", "Hot ashes for trees?", "Hot air for a cool breeze?"])

def test_remove_objective_sents():
    assert remove_objective_sents(["And did they get you to trade Your heros for ghosts?", "Hot ashes for trees?"]) == ['Hot ashes for trees?']


def test_scikitlearn_classifier_exceptions():
    clf = DummyClassifier(strategy="The Ramones") ### :)
    assert_raises(ValueError, clf.fit, [], [])
    assert_raises(ValueError, clf.predict, [])
    assert_raises(ValueError, clf.predict_proba, [])


