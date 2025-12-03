import abc
from abc import abstractmethod

class VectorizedTextBase(metaclass=abc.ABCMeta):

    # 更新数据库
    @abstractmethod
    def store(self, **k):
        pass

    # 在知识库查询
    @abstractmethod
    def query(self, **k):
        pass
