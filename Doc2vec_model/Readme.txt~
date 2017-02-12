Welcome to Doc2vec!
=================


About Problem:
	Use Doc2vec technique to find semantic relationship between news article.With collection of news-article we trained a model.
	In which Each Document(News Article) is represented as a Vector.These vectors give semantic information about document.
	By Infering input vector we get semantic relationship between corpus and source document.

About Code:
	1.) doc2vec.py file train a model from given corpus.

	2.) clean_doc.py file clean document like tokenization,remove stop-words,lemmatization.

	3.) vec_model.py file prepare_data for training and Train model calling doc2vec.py.Actually here you tune the parametres of doc2vec 
		model for better performance.it takes command line arguments:

		vec_model.py [-h] [--stop_words STOP_WORDS] [--lemmatize LEMMATIZE]
                    [--min_count MIN_COUNT] [-w WINDOW_SIZE] [--epoch EPOCH]
                    [--alpha ALPHA] [--min_alpha MIN_ALPHA]
                    [--model_path MODEL_PATH]
                    corpus_path

		ex - python vec_model.py Semantics.json --stop_words True/False --lemmatize True/False --min_count 40 -w 10 --epoch 50 --alpha .025 --min_alpha .025 --model_path model.doc2vec


	4.) evaluate.py file used to evalaute our model. like by finding semantic relationship between documents.

	5.) find_relation.py file process query(source document) and find sematic related documents.

	6.) util.py convert iterator into a list.
