import pandas as pd
import os
import ast
from scipy import spatial
from model.VectorizedTextSimilarityMatching.base import VectorizedTextBase
from model.TextVectorization.base import TextVectorizationBase
from pathlib import Path

class VectorDB(VectorizedTextBase):
    def __init__(self, embedding:TextVectorizationBase):
        self.embedding = embedding
        self.chunks = []

    def store(self, text, savepath: [str,Path]):
        if isinstance(text, str):
            if text == '':
                return
            vector = self.embedding.embed_documents([text])
            df = pd.DataFrame({"text": text, "embedding": vector})
        elif isinstance(text, list):
            if len(text) == 0:
                return
            vector = self.embedding.embed_documents(text)
            df = pd.DataFrame({"text": text, "embedding": vector})
        else:
            raise TypeError('text must be str or list')
        df.to_csv(savepath, mode='a', header=not os.path.exists(savepath), index=False)

    def query(self, text: str, top_n: int,inputpath: str, threshold: float = 0.7):
        if text == '':
            return ['']
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

        # 加载记忆数据
        if not os.path.isfile(inputpath):
            return ['']
        df = pd.read_csv(inputpath)
        row = df.shape[0]
        top_n = min(top_n, row)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

        # 查询
        query_embedding = self.embedding.embed_query(text)
        strings_and_relatednesses = [
            (row["text"], relatedness_fn(query_embedding, row["embedding"]))
            for i, row in df.iterrows()
        ]

        # 计算排行
        strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
        strings, relatednesses = zip(*strings_and_relatednesses)
        i = 0
        for i in range(len(relatednesses)):
            if relatednesses[i] < threshold:
                break
        return strings[:min(i + 1, top_n)], relatednesses[:min(i + 1, top_n)]
