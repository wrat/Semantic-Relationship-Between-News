from clean_doc import clean_doc
from gensim import corpora, models , utils
import gensim


def prepare_corpus(texts,corpus_path,dictionary_path):

        #Construct document-term matrix
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

	dictionary.save(dictionary_path)
	corpora.MmCorpus.serialize(corpus_path, corpus)

def Lda_model(corpus_path,dictionary_path,Num_topics,Passes):


	corpus = corpora.MmCorpus(corpus_path)
	dictionary = gensim.corpora.Dictionary.load(dictionary_path)	

        #Applying the LDA model	
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=Num_topics, id2word = dictionary, passes=Passes)
        return ldamodel


def Lsi_model(model_path,corpus_path,dictionary_path,Num_topics,chunksize,distributed):

	corpus = corpora.MmCorpus(corpus_path)
	dictionary = gensim.corpora.Dictionary.load(dictionary_path)

	#Applying the LSI model
	lsimodel = models.LsiModel(corpus, id2word=dictionary, num_topics=Num_topics, chunksize = chunksize , distributed = distributed)
	lsimodel.save(model_path)
	return lsimodel
