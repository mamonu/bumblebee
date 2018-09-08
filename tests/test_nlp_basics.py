
from nlpfunctions.basicnlp import (
    sent_tokenise,
    get_sentiment_score_VDR,
    get_sentiment_score_TB,
)
from nlpfunctions.utils import *


import numpy as np


def test_basics_sent_tok_hasoutput():

    assert sent_tokenise("this is the end. my only friend. the end") is not None


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
