import pandas as pd
import numpy as np
import os
import ast
from scipy import spatial
from model.VectorizedTextSimilarityMatching.base import VectorizedTextBase
from model.TextVectorization.base import TextVectorizationBase
from model.base import BaseModel

class GraphDB(VectorizedTextBase):
    def __init__(self,embedding: TextVectorizationBase):
        self.embedding = embedding

    def Setllm(self,llm: BaseModel):
        self.llm = llm

    def SetGraphpath(self,Graphpath: str):
        self.Graphpath = Graphpath

    # 把一列表文本或者一段文本储存在savepath的csv里面
    def store(self, text, savepath: str):
        pass

    # 计算与text相似度最高的top_n个文本,来自inputpath的知识库,阈值为threshold
    def query(self, text: str, top_n: int,inputpath: str, threshold: float = 0):
        if text == '':
            return ['']
        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)
        entity = self.llm.think(f"从以下文本中提取主要实体，并以'实体'的格式输出:\n\n{text}")

        # 加载记忆数据
        if not os.path.isfile(inputpath):
            return ['']
        df = pd.read_csv(inputpath)
        row = df.shape[0]
        top_n = min(top_n, row)
        df['embedding'] = df['embedding'].apply(ast.literal_eval)

        # 查询
        query_embedding = self.embedding.embed_query(entity)
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

        strings_entity = strings[:min(i + 1, top_n)]
        data = np.array(pd.read_csv(self.Graphpath)[["实体a","关系","实体b"]])
        output_list = []
        for entityi in strings_entity:
            output_list.extend(["".join([str(j) for j in i]) for i in data[entityi == data[:,0]]])
        return output_list

    # 计算与text相似度最高的top_n个文本,来自inputpath的知识库,阈值为threshold
    def query_(self, text: str, top_n: int, inputpath: str, threshold: float = 0):
        if text.strip() == '':
            return ['']

        relatedness_fn = lambda x, y: 1 - spatial.distance.cosine(x, y)

        # 提取多个实体
        entity_str = self.llm.think(f"从以下文本中提取所有主要实体，并以逗号分隔的'实体1,实体2,...'格式输出:\n\n{text}")
        entities = [e.strip() for e in entity_str.split(',') if e.strip()]

        if not os.path.isfile(inputpath):
            return ['']

        df = pd.read_csv(inputpath)
        if df.empty:
            return ['']

        df['embedding'] = df['embedding'].apply(ast.literal_eval)

        # 加载图谱数据
        graph_df = pd.read_csv(self.Graphpath)
        graph_data = np.array(graph_df[["实体a", "关系", "实体b"]])

        all_output = []

        # 对每个实体分别查询和跳变
        for entity in entities:
            query_embedding = self.embedding.embed_query(entity)
            strings_and_relatednesses = [
                (row["text"], relatedness_fn(query_embedding, row["embedding"]))
                for _, row in df.iterrows()
            ]
            strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
            strings, relatednesses = zip(*strings_and_relatednesses)

            i = 0
            for i in range(len(relatednesses)):
                if relatednesses[i] < threshold:
                    break
            strings_entity = strings[:min(i + 1, top_n)]

            # 跳变图谱
            for entity_text in strings_entity:
                jumps = ["".join(map(str, triple)) for triple in graph_data[graph_data[:, 0] == entity_text]]
                all_output.extend(jumps)

        return list(set(all_output))  # 去重返回

"""
if __name__ == "__main__":
    from model.chatgpt import GPT
    from model.TextVectorization import STEmbedding
    gpt = GPT(OPENAI_API_KEY="",url="api.chatanywhere.com.cn")
    embedding = STEmbedding(modelpath = "../../data/paraphrase-multilingual-MiniLM-L12-v2/")
    graphdb = GraphDB(embedding=embedding)
    graphdb.Setllm(gpt)
    graphdb.SetGraphpath("../../data/KnowledgeGraph/Graph.csv")
    print(graphdb.query(text="物流机器人是什么",top_n=2,threshold=0.8,inputpath="../../data/knowledgeBases/base3.csv"))
"""