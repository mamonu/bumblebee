import pandas as pd
import pytest
from bumblebee.basicnlp import sent_tokenise
from bumblebee.utils import (
    combine_functions,
    output_series,
    combine_2fs,
    word_tokens2string_sentences,
    list2string,
    flattenIrregularListOfLists,
    merge_dfs,
)


# string
mystring = "i do not care. i think that this is bad but maybe not."


def capital_case(x):
    if not isinstance(x, str):
        raise TypeError("Please provide a string argument")
    return x.capitalize()


f_pipeline = combine_functions(capital_case, sent_tokenise)


def test_combine_functions():
    assert sent_tokenise(capital_case(mystring)) == f_pipeline(mystring)


f_pipeline2 = combine_2fs(capital_case, sent_tokenise)


def test_combine_2fs():
    assert sent_tokenise(capital_case(mystring)) == f_pipeline2(mystring)


def test_output_series():
    assert output_series(mystring).all() == (pd.Series(dict(outcome=mystring))).all()


def test_word_tokens2string_sentences():
    assert [
        "i do not care.",
        "i think that this is bad but maybe not.",
    ] == word_tokens2string_sentences(
        [
            ["i", "do", "not", "care."],
            ["i", "think", "that", "this", "is", "bad", "but", "maybe", "not."],
        ]
    )


def test_list2string():
    assert (
        list2string(["i", "think", "that", "this", "is", "bad", "but", "maybe", "not."])
        == "i think that this is bad but maybe not."
    )


def test_flattenIrregularListOfLists():
    assert list(
        flattenIrregularListOfLists(
            [
                ["i", "do", ["not", "care."]],
                ["i", "think", "that", "this", "is", "bad", "but", "maybe", "not."],
            ]
        )
    ) == [
        "i",
        "do",
        "not",
        "care.",
        "i",
        "think",
        "that",
        "this",
        "is",
        "bad",
        "but",
        "maybe",
        "not.",
    ]


def test_merge_dfs():
    assert (
        (
            merge_dfs(
                pd.DataFrame({"A": [1, 2, 3, 4]}), pd.DataFrame({"B": [5, 6, 7, 8]})
            )
            == pd.DataFrame({"A": [1, 2, 3, 4], "B": [5, 6, 7, 8]})
        ).all()
    ).all()
