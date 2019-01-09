import pytest 
from nlpbumblebee.parsing import extract_noun_phrases

testtext = 'The quick brown fox jumps over the lazy dog.'

def test_extract_noun_phrases():
	assert extract_noun_phrases(testtext) == ['quick brown fox jumps', 'lazy dog']