import topic_modelling
from tf_idf  import tf_idf_model
import logging
import sys
import gensim
from clean_doc import clean_doc
from util import listify
import json
from gensim import corpora, models , utils,similarities
from semantics import semantic_relations


source = list()
url_list = list()

@listify
def preprocessing(corpus,stop_words,lemmatize):

	with open(corpus) as data_file:
		data = json.load(data_file)
		for index , line in enumerate(data):
			# Print a status message every 1000th news-feed
		       	if index%1000. == 0.:
			  	print "news %d of %d" % (index, len(data))
		 	tag = line['tag']
			raw_data = line['boilerhtml'].lower()
			url      = line['url']	
			clean_data = clean_doc(raw_data,stop_words=stop_words,lemmatize=lemmatize)
			if(tag != 'corpus'):
				source.append(clean_data)
				continue
			url_list.append(url)
			yield clean_data

def load(model_path,corpus_path,dictionary_path):

	lsi = models.LsiModel.load(model_path)
	corpus = corpora.MmCorpus(corpus_path)
	dictionary = corpora.Dictionary.load(dictionary_path)

	return lsi,corpus,dictionary
		
def train_model(docs,model_path,corpus_path,dictionary_path):

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	topic_modelling.prepare_corpus(docs,corpus_path,dictionary_path)
	lsimodel = topic_modelling.Lsi_model(model_path,corpus_path, dictionary_path,Num_topics,chunksize,distributed)


if __name__ == '__main__':

	docs = preprocessing('Semantics.json',False,False)
	#LSI Parameters
	Num_topics = len(docs)#Decide Number of topic
	chunksize = 1
	distributed = False

	#Save
	corpus_path = 'news_lsi.mm'
	dictionary_path = 'news.dictionary'
	model_path = 'lsi_news.model'
	index_path = 'news.index'

	train_model(docs,model_path,corpus_path,dictionary_path)
	lsi,corpus,dictionary = load(model_path,corpus_path,dictionary_path)

	
	doc = source[1]

	semantic_relations(lsi,corpus,dictionary,doc,url_list,index_path)








