import pytest
import enchant
import re
from enchant.checker import SpellChecker
from nltk import word_tokenize
from nlpfunctions.spellcheck import infer_spaces,find_and_print_errors,correct_text

d = SpellChecker("en_UK","en_US")
testtext='this is a gud beer'



def test_spellcheck_correct_num_errors():
    num_errors, errors = find_and_print_errors(testtext)
    
    assert (num_errors==1)


def test_spellcheck_find_errors():
    num_errors, errors = find_and_print_errors(testtext)
    assert errors==['gud']


def test_spellcheck_correct_errors():
    final=correct_text(testtext)

    assert final == 'this is a @@gud@@ Gus beer'
    

def test_inferspaces():
    textstucktogether ="thereismassesoftextinformation"
    textwithspaces=['there', 'is', 'masses', 'of', 'text', 'information']
    assert infer_spaces(textstucktogether)== " ".join(str(x) for x in textwithspaces)