import abc
from abc import abstractmethod

class TextVectorizationBase(metaclass=abc.ABCMeta):

    # 把一列表数据处理为一列表向量
    @abstractmethod
    def embed_documents(self, documents: list):
        pass

    # 把一段数据处理为一个向量
    @abstractmethod
    def embed_query(self, text: str):
        pass
