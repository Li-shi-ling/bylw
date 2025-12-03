from .base.TextVectorizationBase import TextVectorizationBase
from glove import Glove
import numpy as np

class GloVeVectorizer(TextVectorizationBase):
    def __init__(self, glove_model_path):
        self.glove = Glove.load(glove_model_path)
        self.embedding_dim = self.glove.no_components

    def get_word_embedding(self, word):
        try:
            return self.glove.word_vectors[self.glove.dictionary[word]]
        except KeyError:
            return np.zeros(self.embedding_dim)

    def embed_documents(self, documents: list):
        return [self.embed_query(doc) for doc in documents]

    def embed_query(self, text: str):
        words = text.split()
        embeddings = [self.get_word_embedding(word) for word in words]
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_dim)
