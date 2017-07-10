'''
    Doc2Vec wrapper

    A simple wrapper so that gensim's doc2vec routines
    can integrate with sklearn pipelines.

    This assumes:
        - python 2.7
        - sklearn 0.18.1
        - gensim 1.0.1

    This wrapper only contains the class Doc2VecTransformer
    which has fit, fit_transform and transform methods.

    Note, because doc2vec's infer_vectors method does not
    return the same vectors as produced in model build for
    documents used to train the model, fit_transform is not
    just fit followed by transform.
'''

import re
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from gensim.models.doc2vec import TaggedDocument
from gensim.models.doc2vec import Doc2Vec

class Doc2VecTransformer(BaseEstimator, TransformerMixin):

    def default_tokeniser(self, text):
        ''' Basic tokeniser to use as a default '''
        return re.findall(r"\S+", text)

    def labeller_id(self, ID, text):
        ''' Converts data to the TaggedDocument format gensim expects '''
        return TaggedDocument(self.tokeniser(text), [ID])

    def __init__(self, size=100, seed=1, min_count=5, window=5, iter=5, tokeniser='auto'):
        ''' Initialise variables and select tokeniser '''
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
        ''' Fit the doc2vec model '''

        # Convert documents into TaggedDocument format using the
        # index of the document in the array as the label
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
        ''' 
        Fit the doc2vec model and transform the original text to vectors.
        This is not just fit followed by transform as retrieving the vectors
        for the training documents is more accurate than inferring the vectors
        '''

        # Fit the model
    	self.fit(X)

        # Return the document vectors
    	return self.model.docvecs.doctag_syn0

    def transform(self, X, y=None):
        ''' Transform new documents '''

        # Tokenise the text
        tokenised = map(self.tokeniser,  X)

        # Infer vectors for tokenised text
        transformed = np.matrix(map(lambda x: self.model.infer_vector(x, steps=25), tokenised))
        return transformed
