import pytest
from nlpbumblebee.stringutils import (
    string_matching_knuth_morris_pratt,
    kmp_prefix_function,
    firstoccurence,
    lastoccurence,
)


def test_string_matching_KMP():
    haystack = "ababbababa"
    needle = "aba"
    assert string_matching_knuth_morris_pratt(haystack, needle) == [0, 5, 7]


def test_stringisnotmatching_KMP():
    haystack = "ababbababa"
    needle = "lotr"
    assert string_matching_knuth_morris_pratt(haystack, needle) == []


def test_compute_prefix_function():
    assert kmp_prefix_function("abc") == [0, 0, 0]


def test_firstoccurence():
    haystack = "ababbababa"
    needle = "aba"
    assert firstoccurence(haystack, needle) == 0


def test_nofirstoccurence():
    haystack = "ababbababa"
    needle = "omg"
    assert firstoccurence(haystack, needle) == -5000


def test_lastoccurence():
    haystack = "ababbababa"
    needle = "aba"
    assert lastoccurence(haystack, needle) == 7
