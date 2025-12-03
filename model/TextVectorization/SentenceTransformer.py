from model.TextVectorization.base import TextVectorizationBase
from sentence_transformers import SentenceTransformer
from termcolor import colored

class STEmbedding(TextVectorizationBase):
    def __init__(self,**kwargs):
        try:
            self.model = SentenceTransformer(kwargs.get("modelpath"))
        except:
            print(colored('Sentence Transformer 模型加载失败！', 'red'))

    def embed_documents(self, documents: list):
        return list(self.model.encode(documents).tolist())

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()
