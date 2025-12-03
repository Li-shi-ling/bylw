import abc
from abc import abstractmethod

class EmotionBase(metaclass=abc.ABCMeta):

    # 根据text返回感情
    @abstractmethod
    def think(self, text: str):
        pass
