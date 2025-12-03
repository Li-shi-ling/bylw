import yaml
import time
from model.chatgpt import GPT
from model.chatgml import GML
from model.TextVectorization.SentenceTransformer import STEmbedding
from model.VectorizedTextSimilarityMatching.VectorDB import VectorDB
from model.VectorizedTextSimilarityMatching.GraphDB import GraphDB
from dialogue.DatabaseMounting import DatabaseMounting
from dialogue.KnowledgeGraph import KnowledgeGraph

#*******************************************************
#代码作者：lishiling 完成时间:2024.9.3
#*******************************************************

import warnings
warnings.filterwarnings("ignore")

with open('./config/base.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
result["model"] = globals()[result["model"]](**(result.get(result["model"],{})))
result["TextVectorization"] = globals()[result["TextVectorization"]](**(result.get(result["TextVectorization"],{})))
result["VectorizedTextSimilarityMatching"] = globals()[result["VectorizedTextSimilarityMatching"]](result["TextVectorization"])
dialogue = globals()[result["dialogue"]](**result)

while(True):
    print(f"\nAI:{dialogue.ask(input('user:'))}\n")
