from gensim import corpora, models , utils,similarities

def semantic_relations(lsi,corpus,dictionary,source_doc,url_list,index_path):

	vec_bow = dictionary.doc2bow(source_doc)
	vec_lsi = lsi[vec_bow]

	index = similarities.MatrixSimilarity(lsi[corpus])
	index.save(index_path)
	index = similarities.MatrixSimilarity.load(index_path)
	
	sims = index[vec_lsi]
	
	#Short similar document
	sims = sorted(enumerate(sims), key=lambda item: -item[1])


	for sim in  sims:
		print(url_list[sim[0]])
