from textblob import TextBlob
from stop_words import get_stop_words
from nltk.stem.wordnet import WordNetLemmatizer

def clean_doc(doc,stop_words = None , lemmatize = None):

	blob = TextBlob(doc)
        tokens = blob.words # here tokens are words of document
        
	# remove stop words from tokens
	if(stop_words):		
		# create English stop words list
		en_stop = get_stop_words('en')
		stopped_tokens = [i for i in tokens if not i in en_stop]
		tokens = stopped_tokens

	if(lemmatize):
        	# Create WordNet_Lemmatizer of class WordNetLemmatizer
        	lemmatizer = WordNetLemmatizer()      
        	# Lemmatize token
		leammatized_tokens = [lemmatizer.lemmatize(i) for i in tokens]
		tokens = leammatized_tokens
 
	return tokens
