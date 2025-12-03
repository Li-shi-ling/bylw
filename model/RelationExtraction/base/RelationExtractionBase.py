import abc
from abc import abstractmethod

class RelationExtractionBase(metaclass=abc.ABCMeta):

    # 实体关系提取
    @abstractmethod
    def extraction(self, documents: list):
        pass
