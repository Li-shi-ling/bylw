import abc
from abc import abstractmethod

class DialogueBase(metaclass=abc.ABCMeta):

    # 一键式调用
    @abstractmethod
    def ask(self, text: str):
        pass
