from clean_doc import clean_doc
from gensim import corpora, models

def tf_idf_model(docs):
	
        texts = [clean_doc(doc) for doc in docs]

        #Construct document-term matrix
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

	#Applying tfidf model
	tfidf = models.TfidfModel(corpus)
	corpus_tfidf = tfidf[corpus]
	
	#Create a dicitionary for getting tf-idf scores of words w.r.t corpus
	tf_idf_score = {dictionary.get(id): value for doc in corpus_tfidf for id, value in doc}

	return corpus_tfidf , tf_idf_score
