import enchant
from enchant.checker import SpellChecker
import re
from nltk import word_tokenize
from math import log
from wordninja import split as wnsplit


def infer_spaces(s):
    """ Probabilistically split concatenated words using NLP based on English Wikipedia uni-gram frequencies. """
    ## Ref:   https://stackoverflow.com/questions/8870261/how-to-split-text-without-spaces-into-list-of-words

    return " ".join(str(x) for x in wnsplit(s))


d = SpellChecker("en_UK", "en_US")


def find_and_print_errors(text):
    errors = list(
        set(
            [
                word
                for word in word_tokenize(text)
                if d.check(word) is False and re.match("^[a-zA-Z ]*$", word)
            ]
        )
    )
    num_errors = len(errors)

    return (num_errors, errors)


def correct_text(text):
    d.set_text(text)

    for err in d:

        if len(err.suggest()) > 0:
            err.replace("@@%s@@ %s" % (err.word, err.suggest()[0]))
        else:
            err.replace("**%s**" % (infer_spaces(err.word)))

    final = d.get_text()

    return final
