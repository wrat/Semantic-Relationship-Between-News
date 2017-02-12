from clean_doc import clean_doc
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import gensim.models.doc2vec
import gensim  
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"


class LabeledLineSentence(object):
    def __init__(self, data):
       self.data = data

    def __iter__(self):
        for text,url in  self.data:				
            yield LabeledSentence(words = text , tags = url)

def doc2vec(data , alpha , window , min_alpha , min_count , epoch , model_path):

	#Labeled Sentences
	sentences = LabeledLineSentence(data)

	model = Doc2Vec(alpha = alpha , window = window , min_alpha = min_alpha , min_count = min_count , workers=cores)  # use fixed  learning rate

	model.build_vocab(sentences)

	for epoch_count in range(epoch):# Change for good performance
	    model.train(sentences)
	    model.alpha -= 0.002  # decrease the learning rate
	    model.min_alpha = model.alpha  # fix the learning rate, no decay
	
	# store the model to mmap-able files
	model.save(model_path)
