from model.VectorizedTextSimilarityMatching.GraphDBforweb import GraphDBforweb
from model.TextVectorization.SentenceTransformer import STEmbedding
from flask import Flask, render_template, request, jsonify
from dialogue.KnowledgeGraphWeb import KnowledgeGraphWeb
from model.chatdeepseek import DeepSeek
from util.tool import Getentity
from model.chatgpt import GPT
from model.chatgml import GML
import warnings
import yaml
import time
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR, 2 = WARNING, 1 = INFO, 0 = DEBUG

warnings.filterwarnings("ignore")

with open('../config/GWbase.yaml', 'r', encoding='utf-8') as f:
    result = yaml.load(f.read(), Loader=yaml.FullLoader)
result["model"] = globals()[result["model"]](**(result.get(result["model"],{})))
result["VectorizedTextSimilarityMatching"] = globals()[result["VectorizedTextSimilarityMatching"]](**(result.get(result["VectorizedTextSimilarityMatching"],{})))
dialogue = globals()[result["dialogue"]](**result)

# while(True):
#     print(f"\nAI:{dialogue.ask(input('user:'))}\n")

app = Flask(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

with open('data/graph_triples.json') as f:
    triples_data = json.load(f)

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/vector-db')
def vector_db():
    vector_path = os.path.join(DATA_DIR, 'vector_data.json')
    with open(vector_path, encoding='utf-8') as f:
        vectors = json.load(f)
    return render_template('vector_db.html', vectors=vectors)

@app.route('/graph')
def graph():
    return render_template('graph.html')

@app.route('/get-triples', methods=['POST'])
def get_triples():
    keyword = request.json.get('keyword', '')
    if keyword == '':
        return jsonify(triples_data)
    datas = Getentity(keyword)
    if len(datas) == 0:
        return jsonify([[keyword,"没找到","相关实体"]])
    # filtered = [t for t in triples_data if keyword in t[0] or keyword in t[2]]
    filtered = [data for data in datas]
    return jsonify(filtered)

@app.route('/chat', methods=['POST'])
def chat_api():
    data = request.get_json()
    question = data.get('question', '')

    answer = dialogue.ask(question)

    triples = dialogue.get_entity_triplet()
    # answer = """
    # # include <stdio.h>
    #
    # int main(){
    #     return 0;
    # }
    # """
    # triples = []

    return jsonify({"answer": answer, "triples": triples})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
