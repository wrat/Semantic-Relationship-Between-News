from argparse import ArgumentParser
from gensim.models import Doc2Vec

def Load_model(model_path):

	#load the model back
	model = Doc2Vec.load(model_path)	
	return model


def most_similar_document(model,document,topn=5):

	infer_vector = model.infer_vector(document)
	similar_documents = model.docvecs.most_similar([infer_vector], topn=topn)#List Of Tuples

	url_list = list()
	for similar_document in similar_documents:
		url_list.append(similar_document[0])
	
	return url_list

def parse_args():

    parser = ArgumentParser(
        description=('Evaluate a Document-vector-space model on a'
                     'Semantic-Relationship-Task'))

    parser.add_argument('model_path',  help=('Path to Trained Doc2vec Model'))

    parser.add_argument('Source_documents_path',help=('Path For Input Source Document For which we want to find semantic relationship'))

    return parser.parse_args()
