from textblob import TextBlob
import nltk

#for various syntax based filtering options

def extract_noun_phrases(text):
	stop_words = set(nltk.corpus.stopwords.words("english")) # to remove stopwords
	blob = TextBlob(text) # create a textblob object 
	np = blob.noun_phrases # selecting all noun-phrases
	np = [w for w in np if len(w)>=3] # remove nounphrses that are less than 3 characters
	nounps = [
		word.lower()
		for word in np
		if word.lower() not in stop_words]
	return nounps