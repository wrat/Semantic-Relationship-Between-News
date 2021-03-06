import Evaluate
import logging
from clean_doc import clean_doc
import json

#Return a dictionary mapping each_document to semactic related documents
def Semantic_relation(source_file,model_path,topn,stop_words,lemmatize):
	
	model = Evaluate.Load_model(model_path)

	Semantic_relations = dict()
	with open(source_file) as data_file:
		data = json.load(data_file)
		for index , document in enumerate(data):
			raw_data = document['boilerhtml']
			url      = document['url']	
			clean_data = clean_doc(raw_data,stop_words=stop_words,lemmatize=lemmatize)
			
			url_list = Evaluate.most_similar_document(model,clean_data,topn = topn)

			Semantic_relations[url] = url_list

	return Semantic_relations

if __name__ == '__main__':

	stop_words = False
	lemmatize = True
	source_file = ''
	model_path  = ''
	topn = 5
	Semantic_relations = Semantic_relation(source_file,model_path,topn,stop_words,lemmatize)
