import yaml
import time
from model.chatgpt import GPT
from model.chatgml import GML
from model.chatdeepseek import DeepSeek
from model.TextVectorization.SentenceTransformer import STEmbedding
from model.VectorizedTextSimilarityMatching import VectorDB,GraphDB,GraphDBforweb
from dialogue import DatabaseMounting,KnowledgeGraph,KnowledgeGraphWeb

#*******************************************************
#代码作者：lishiling 完成时间:2025.4.20
#*******************************************************

import warnings
warnings.filterwarnings("ignore")

with open('./config/GWbase.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
result["model"] = globals()[result["model"]](**(result.get(result["model"],{})))
result["VectorizedTextSimilarityMatching"] = globals()[result["VectorizedTextSimilarityMatching"]](**(result.get(result["VectorizedTextSimilarityMatching"],{})))
dialogue = globals()[result["dialogue"]](**result)

while(True):
    print(f"\nAI:{dialogue.ask(input('user:'))}\n")
