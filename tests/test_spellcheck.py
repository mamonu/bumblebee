import pytest
import enchant
import re
from enchant.checker import SpellChecker
from nltk import word_tokenize

d = SpellChecker("en_UK","en_US")
testtext='this is a gud beer'



def test_find_num_errors():
    errors=(list(set([word for word in word_tokenize(testtext) if d.check(word) is False and re.match('^[a-zA-Z ]*$',word)] )))
    num_errors=len(errors)
    assert (num_errors==1)


def test_find_errors():
    errors=(list(set([word for word in word_tokenize(testtext) if d.check(word) is False and re.match('^[a-zA-Z ]*$',word)] )))
    num_errors=len(errors)
    assert errors==['gud']


def test_correct_text():
    d.set_text(testtext)
    for err in d:
        if len(err.suggest())>0: 
            err.replace((err.suggest()[4])) 
    final = d.get_text()
    print(final)

    assert final == 'this is a god beer'