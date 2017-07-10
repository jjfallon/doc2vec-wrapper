import re
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

class Doc2VecTransformer(BaseEstimator, TransformerMixin):

    def default_tokeniser(self, text):
		return re.findall(r"\S+", text)

    def labeller_id(self, ID, text):
		return TaggedDocument(self.tokeniser(text), [ID])

    def __init__(self, size=100, seed=1, min_count=5, window=5, iter=5, tokeniser='auto'):
		self.size = size
		self.seed = seed
		self.min_count = min_count
		self.window = window
		self.iter = iter

		if( tokeniser == 'auto'):
			self.tokeniser = self.default_tokeniser
		else:
			self.tokeniser = tokeniser

    def fit(self, X, y=None):
        docs = map(lambda x: self.labeller_id(x[0], x[1]), enumerate(X))

        # Build model
        self.model = Doc2Vec(documents = docs,
        				size = self.size,
        				seed = self.seed,
        				min_count = self.min_count,
        				#max_vocab_size = None,
        				window = self.window,
        				iter = self.iter)

        # Free up memory
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
        return self

    def fit_transform(self, X, y=None):
    	self.fit(X)
    	return self.model.docvecs.doctag_syn0

    def transform(self, X, y=None):
        tokenised = map(self.tokeniser,  X)
        transformed = np.matrix(map(lambda x: self.model.infer_vector(x, steps=25), tokenised))
        return transformed
