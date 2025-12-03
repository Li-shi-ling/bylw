from langchain.schema import HumanMessage,BaseMessage,AIMessage,SystemMessage
from typing import List, Dict, Any
from dateutil.parser import parse
from pydub.playback import play
from pydub import AudioSegment
from termcolor import colored
import numpy as np
import datetime
import requests
import json
import os
import re

# 获取文件的行数
def count_lines_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        return sum(1 for line in file)

# 统计所有文件的行数
def count_lines_in_directory(directory_path):
    total_lines = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                total_lines += count_lines_in_file(file_path)
            except Exception as e:
                print(f"Could not read file {file_path}: {e}")
    return total_lines

# 统计py文件的代码数量
def count_lines_in_directory_py(directory_path):
    total_lines = 0
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    total_lines += count_lines_in_file(file_path)
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")
    return total_lines

#制造有时间的消息
def make_message(text: str,HumanMessageT:bool = True):
    if HumanMessageT:
        data = {
            "msg": text,
            "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return HumanMessage(content=json.dumps(data, ensure_ascii=False))
    else:
        data = {
            "msg": text,
            "time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        return AIMessage(content=json.dumps(data, ensure_ascii=False))

#返回最后一条消息到现在的小时数
def message_period_to_now(message: BaseMessage):
    last_time = json.loads(message.content)['time']
    last_time = parse(last_time)
    now_time = parse(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    duration = (now_time - last_time).total_seconds() / 3600
    return duration

#将话分割,提取第一句
def get_first_sentence(text: str):
    sentences = re.findall(r'.*?[~。！？…]+', text)
    if len(sentences) == 0:
        return '', text
    first_sentence = sentences[0]
    after = text[len(first_sentence):]
    return first_sentence, after

#加载人设
def load_prompt(file_path: str):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(colored(f'人设文件加载成功！({file_path})', 'green'))
    except:
        print(colored(f'人设文件: {file_path} 不存在', 'red'))
    return system_prompt

#加载记忆
def load_memory(file_path: str, waifuname):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            memory = f.read()
        if os.path.exists(f'./memory/{waifuname}.csv'):
            print(colored(f'记忆数据库存在，不导入记忆', 'yellow'))
            return ''
        else:
            chunks = memory.split('\n\n')
            print(colored(f'记忆导入成功！({len(chunks)} 条记忆)', 'green'))
    except:
        print(colored(f'记忆文件文件: {file_path} 不存在', 'red'))
    return memory

#将mp3转变为wav
def mp32wav(file_path):
    audio = AudioSegment.from_file(file_path, format='mp3')
    audio.export(file_path.replace(".mp3",".wav"), format='wav')

#播放mp3或wav文件
def play_audio(file_path,type = "mp3"):
    song = None
    if type == "mp3":
        song = AudioSegment.from_file(file_path, format=type)
    elif type == "wav":
        song = AudioSegment.from_file(file_path, format=type)
    if not song is None:
        play(song)

def PromptGenerate(messages,user_name="user",AI_name="AI"):
    prompt = ""
    for mes in messages:
        if isinstance(mes, HumanMessage):
            prompt += f'{user_name}: {mes.content}\n\n'
        elif isinstance(mes, SystemMessage):
            prompt += f'System Information: {mes.content}\n\n'
        elif isinstance(mes, AIMessage):
            prompt += f'{AI_name}: {mes.content}\n\n'
    return prompt

class KnowledgeGraphAPIError(Exception):
    pass

def validate_mention_name(mention_name: str) -> None:
    if not isinstance(mention_name, str):
        raise ValueError("实体名称必须是字符串")
    if not mention_name.strip():
        raise ValueError("实体名称不能为空")
    if len(mention_name) > 100:
        raise ValueError("实体名称过长，最大长度为100个字符")

def validate_api_response(response_data: Dict[str, Any]) -> None:
    if not isinstance(response_data, dict):
        raise KnowledgeGraphAPIError("API响应格式无效，预期为JSON对象")

    if "message" in response_data and response_data["message"] != "success":
        raise KnowledgeGraphAPIError(f"API返回错误: {response_data.get('message', '未知错误')}")

    if "data" not in response_data:
        raise KnowledgeGraphAPIError("API响应中缺少'data'字段")

def process_avp_data(avp_data: List[List[str]]) -> List[str]:
    if not isinstance(avp_data, list):
        return []

    processed = []
    for item in avp_data[:10]:  # 限制最多10个属性
        if (isinstance(item, list) and len(item) >= 2 and
                isinstance(item[0], str) and isinstance(item[1], str)):
            processed.append(f"{item[0]}:{item[1]}")
    return processed

def Getentity(mention_name: str) -> List[List]:
    try:
        # 输入验证
        validate_mention_name(mention_name)

        # API请求
        response = requests.get(
            f"https://api.ownthink.com/kg/knowledge?entity={mention_name}",
            timeout = 10  # 添加超时
        )
        response.raise_for_status()  # 检查HTTP错误

        # 响应解析
        try:
            data = json.loads(response.text)
        except json.JSONDecodeError:
            raise KnowledgeGraphAPIError("API返回的不是有效JSON")

        # 响应验证
        validate_api_response(data)
        kg_data = data["data"]

        # 数据处理
        outputdata = []
        entity = kg_data.get("entity")
        avps = kg_data.get("avp",None)

        if avps is None:
            return []

        for avp in avps:
            outputdata.append([entity,avp[0],avp[1]])

        return outputdata

    except requests.RequestException as e:
        raise KnowledgeGraphAPIError(f"API请求失败: {str(e)}")


def convert_chinese_punctuation_to_english(text: str) -> str:
    """
    将文本中的中文标点替换为对应的英文标点。
    未在映射表中的标点会被“直接去掉”或保留（视具体实现）。
    """
    # 映射表：中文标点 -> 英文标点
    cn_to_en = {
        '。': '.',
        '，': ',',
        '、': ',',
        '；': ';',
        '：': ':',
        '？': '?',
        '！': '!',
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '《': '<',
        '》': '>',
        '“': '"',
        '”': '"',
        '‘': "'",
        '’': "'",
        '—': '-',   # 或者 “—” -> "--"
        '–': '-',   # en dash
        '…': '...',  # 或者用三个点
        '「': '"',
        '」': '"',
        '『': '"',
        '』': '"',
        '—': '-',
    }
    pattern = '[' + ''.join(re.escape(k) for k in cn_to_en.keys()) + ']'

    def _repl(match: re.Match) -> str:
        ch = match.group(0)
        return cn_to_en.get(ch, '')

    result = re.sub(pattern, _repl, text)
    return result

def find_text_intervals(text, extraction_text):
    """
    在文本中查找提取文本的位置
    """
    start_pos = text.find(extraction_text)
    if start_pos == -1:
        return None
    end_pos = start_pos + len(extraction_text)
    return {"start_pos": start_pos, "end_pos": end_pos}


def align_extractions(data):
    """
    根据文本内容为每个提取项设置alignment_status和char_interval
    """
    text = data["text"]

    for extraction in data["extractions"]:
        extraction_text = extraction["extraction_text"]

        # 查找文本区间
        char_interval = find_text_intervals(text, extraction_text)

        if char_interval:
            extraction["char_interval"] = char_interval
            extraction["alignment_status"] = "match_exact"
        else:
            extraction["char_interval"] = None
            extraction["alignment_status"] = None

    return data
