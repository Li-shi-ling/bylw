import json
import requests
from model.base.BaseModel import BaseModel
from typing import List, Dict, Any
from model.VectorizedTextSimilarityMatching.base.VectorizedTextBase import VectorizedTextBase

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

def Getentity(mention_name: str):
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
        desc = kg_data.get("desc")
        avp = kg_data.get("avp")

        if isinstance(entity, str) and entity.strip():
            outputdata.append(entity.strip())

        if isinstance(desc, str) and desc.strip():
            outputdata.append(desc.strip())

        if isinstance(avp, list):
            outputdata.extend(process_avp_data(avp))

        # 数据处理
        outputdata2 = []
        entity = kg_data.get("entity")
        avps = kg_data.get("avp",None)

        if avps is None:
            return []

        for avp in avps:
            outputdata2.append([entity,avp[0],avp[1]])

        return outputdata, outputdata2

    except requests.RequestException as e:
        raise KnowledgeGraphAPIError(f"API请求失败: {str(e)}")

class GraphDBforweb(VectorizedTextBase):
    def __init__(self, url: str):
        if not isinstance(url, str) or not url.strip():
            raise ValueError("URL不能为空")
        self.url = url
        self.llm = None

    def Setllm(self, llm: BaseModel):
        if not hasattr(llm, 'think'):
            raise ValueError("LLM必须实现think方法")
        self.llm = llm

    def store(self, text, savepath: str):
        pass

    def query(self, text: str, top_n: int, inputpath: str, threshold: float = 0):
        if not isinstance(text, str) or not text.strip():
            return ['']

        if not isinstance(top_n, int) or top_n <= 0:
            raise ValueError("top_n必须是正整数")

        if self.llm is None:
            raise ValueError("LLM未设置，请先调用Setllm方法")

        try:
            # 提取实体
            entity_response = self.llm.think(
                f"从以下文本中提取主要实体，并以'实体1,实体2,...'的格式输出:\n\n{text}"
            )

            if not isinstance(entity_response, str):
                raise ValueError("LLM返回的实体不是字符串")

            entitys = [e.strip() for e in entity_response.split(",") if e.strip()]

            # 获取每个实体的信息
            outputdata = []
            outputdata2 = []
            for entity in entitys[:top_n]:  # 限制最多处理top_n个实体
                try:
                    entity_info, entity_triplet = Getentity(entity)
                    outputdata.extend(entity_info)
                    outputdata2.extend(entity_triplet)
                except (ValueError, KnowledgeGraphAPIError) as e:
                    print(f"获取实体'{entity}'信息失败: {str(e)}")
                    continue

            return outputdata, outputdata2

        except Exception as e:
            print(f"查询知识图谱失败: {str(e)}")
            return ['']
