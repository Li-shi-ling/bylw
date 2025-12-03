import abc
from abc import abstractmethod

class BaseModel(metaclass=abc.ABCMeta):

    # 把一列表文本或者一段文本储存在savepath的csv里面
    @abstractmethod
    def store(self, text, savepath: str):
        pass

    # 计算与text相似度最高的top_n个文本,来自inputpath的知识库,阈值为threshold
    @abstractmethod
    def query(self, text: str, top_n: int,inputpath: str, threshold: float):
        pass
