import enchant
from enchant.checker import SpellChecker
import re
from nltk import word_tokenize
from math import log

# Build a cost dictionary, assuming Zipf's law and cost = -math.log(probability).
words = open("words-by-freq125K.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(words)))) for i,k in enumerate(words))
maxword = max(len(x) for x in words)



def infer_spaces(s):
    """Uses dynamic programming to infer the location of spaces in a string
    without spaces."""

    # Find the best match for the i first characters, assuming cost has
    # been built for the i-1 first characters.
    # Returns a pair (match_cost, match_length).
    def best_match(i):
        candidates = enumerate(reversed(cost[max(0, i-maxword):i]))
        return min((c + wordcost.get(s[i-k-1:i], 9e999), k+1) for k,c in candidates)

    # Build the cost array.
    cost = [0]
    for i in range(1,len(s)+1):
        c,k = best_match(i)
        cost.append(c)

    # Backtrack to recover the minimal-cost string.
    out = []
    i = len(s)
    while i>0:
        c,k = best_match(i)
        assert c == cost[i]
        out.append(s[i-k:i])
        i -= k

    return " ".join(reversed(out))



d = SpellChecker("en_UK","en_US")



def find_and_print_errors(text):
    errors=(list(set([word for word in word_tokenize(text) if d.check(word) is False and re.match('^[a-zA-Z ]*$',word)] )))
    num_errors=len(errors)
    #print("num_errors:",num_errors)
    #print(errors)
    
    return(num_errors,errors)



def correct_text(text):
    d.set_text(text)

    for err in d:

        if len(err.suggest())>0: 
            err.replace("@@%s@@ %s" % (err.word, err.suggest()[0])) 
        else:
            err.replace("**%s**" % (infer_spaces(err.word)) )

    final = d.get_text()

    return (final)










