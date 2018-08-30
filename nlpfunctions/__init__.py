from nlpfunctions import *
import nltk
import pandas as pd 
import numpy as np 
import os 
import string
from math import log
import enchant
from enchant.checker import SpellChecker
import re


# wordcost dictionary, assuming Zipf's law and cost = -math.log(probability).
#useful for the spellcheck submodule
#loaded here to avoid relative path problems
wordsfile = open("words-by-freq125K.txt").read().split()
wordcost = dict((k, log((i+1)*log(len(wordsfile)))) for i,k in enumerate(wordsfile))
maxword = max(len(x) for x in wordsfile)