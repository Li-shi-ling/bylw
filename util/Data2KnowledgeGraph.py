from model.VectorizedTextSimilarityMatching.VectorDB import VectorDB
from model.TextVectorization.SentenceTransformer import STEmbedding
from typing import List, Dict, Any
from docx import Document
from pathlib import Path
import pandas as pd
import numpy as np
import http.client
import textwrap
import yaml
import json
import os

def ask(text, OPENAI_API_KEY):
    """向LLM API发送请求并验证响应格式"""
    type_model = "gpt-3.5-turbo"
    url = "api.chatanywhere.com.cn"
    headers = {
        'Authorization': 'Bearer ' + OPENAI_API_KEY,
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection(url)
        payload = json.dumps({
            "model": type_model,
            "messages": [{
                'role': 'user',
                'content': f"从以下文本中提取实体及其关系，并以'实体1 - 关系 - 实体2'的格式输出:\n\n{text}"
            }]
        })
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        json_data = json.loads(data)

        # 验证响应格式
        if not json_data.get('choices') or len(json_data['choices']) == 0:
            raise ValueError("API response missing 'choices' field")
        if not json_data['choices'][0].get('message', {}).get('content'):
            raise ValueError("API response missing message content")

        return json_data
    except (json.JSONDecodeError, http.client.HTTPException) as e:
        raise ValueError(f"API request failed: {str(e)}")


def validate_kg_line(line):
    """严格验证知识图谱行格式"""
    if not line.strip() or line.count('-') < 2:
        return None

    # 处理实体名称中可能包含的短横线
    parts = []
    temp_parts = line.split('-')
    # 确保关系是中间部分
    if len(temp_parts) >= 3:
        entity1 = temp_parts[0].strip()
        relation = temp_parts[1].strip()
        entity2 = '-'.join(temp_parts[2:]).strip()
        parts = [entity1, relation, entity2]

    return parts if len(parts) == 3 else None


def tocsv(outputpath, data_list):
    """将LLM输出转换为CSV，严格验证格式"""
    with open(outputpath, "w", encoding='utf-8') as f:
        f.write(",".join(["实体a", "关系", "实体b"]) + "\n")
        for data in data_list:
            content = data['choices'][0]['message']['content']
            for line in content.split("\n"):
                validated = validate_kg_line(line)
                if validated:
                    f.write(",".join(validated) + "\n")


def read_docx(file_path):
    """读取docx文件内容"""
    try:
        doc = Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())
    except Exception as e:
        raise ValueError(f"Failed to read docx file: {str(e)}")

def split_long_text(text: str, max_length: int = 3000) -> List[str]:
    """
    将长文本分割成不超过max_length的块
    尽量在段落边界处分割
    """
    if len(text) <= max_length:
        return [text]

    chunks = []
    paragraphs = text.split('\n')
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        if current_length + len(para) + 1 > max_length and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_length = 0

        current_chunk.append(para)
        current_length += len(para) + 1  # +1 for the newline

    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    return chunks


def process_document(file_path: Path, api_key: str, max_chunk_size: int = 3000) -> List[Dict[str, Any]]:
    """
    处理单个文档，自动分块并请求API
    """
    try:
        document_text = read_docx(file_path)
        if not document_text.strip():
            print(f"Warning: Empty document {file_path}")
            return []

        chunks = split_long_text(document_text, max_chunk_size)
        results = []

        for i, chunk in enumerate(chunks):
            try:
                print(f"Processing chunk {i + 1}/{len(chunks)} of {file_path.name}")
                result = ask(chunk, api_key)
                results.append(result)
            except Exception as e:
                print(f"Error processing chunk {i + 1} of {file_path.name}: {str(e)}")
                continue

        return results
    except Exception as e:
        print(f"Error processing document {file_path}: {str(e)}")
        return []


def remove_duplicate_entries(csv_path: Path) -> None:
    """
    对生成的CSV文件进行去重处理
    """
    try:
        df = pd.read_csv(csv_path)
        initial_count = len(df)
        df.drop_duplicates(inplace=True)
        final_count = len(df)

        if initial_count != final_count:
            df.to_csv(csv_path, index=False)
            print(f"Removed {initial_count - final_count} duplicate entries")
    except Exception as e:
        print(f"Error removing duplicates: {str(e)}")


if __name__ == "__main__":
    try:
        config_path = Path("../config/GWbase.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            OPENAI_API_KEY = yaml.safe_load(f)["GPT"]["OPENAI_API_KEY"]

        tmp_output_path = Path("../data/Gdata/Graph.csv")
        output_path = Path("../data/KnowledgeGraph/base.csv")
        files_dir = Path("../data/datafrom/word")

        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)

        datas = []
        for file_path in files_dir.glob("*.docx"):
            file_results = process_document(file_path, OPENAI_API_KEY)
            datas.extend(file_results)

        # 生成临时CSV
        tocsv(tmp_output_path, datas)

        # 去重处理
        remove_duplicate_entries(tmp_output_path)

        # 向量化处理
        Embedding = STEmbedding(modelpath="../data/paraphrase-multilingual-MiniLM-L12-v2/")
        vectordb = VectorDB(Embedding)
        df = pd.read_csv(tmp_output_path)
        rag_input = [i for i in np.unique(df[["实体a"]].values.reshape(-1))]
        vectordb.store(rag_input, output_path)

    except Exception as e:
        print(f"Main execution failed: {str(e)}")
        raise