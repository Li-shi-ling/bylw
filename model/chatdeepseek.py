from langchain.schema import HumanMessage, SystemMessage, AIMessage
from model.base.BaseModel import BaseModel
from util import PromptGenerate
import http.client
import logging
import json


class DeepSeekChat():
    def __init__(self, DEEPSEEK_API_KEY, type_model="deepseek-chat"):
        self.DEEPSEEK_API_KEY = DEEPSEEK_API_KEY
        self.type_model = type_model
        self.headers = {
            'Authorization': f'Bearer {self.DEEPSEEK_API_KEY}',
            'Content-Type': 'application/json'
        }
        self.api_host = "api.deepseek.com"

    def chat(self, inputtext):
        for i in range(5):
            try:
                out_text = self.get_ai_response(inputtext)
            except Exception as e:
                logging.error(f"DeepSeek API请求出错: {e}")
                continue
            break
        else:
            out_text = "请求DeepSeek API时出现错误"
        return out_text

    def get_ai_response(self, text):
        logging.info(f"请求DeepSeek的输入文本: {text}")
        conn = http.client.HTTPSConnection(self.api_host)

        payload = json.dumps({
            "model": self.type_model,
            "messages": [{"role": "user", "content": text}],
            "temperature": 0.7,
            "max_tokens": 2000
        })

        conn.request("POST", "/v1/chat/completions", payload, self.headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        json_data = json.loads(data)

        if "choices" not in json_data or len(json_data["choices"]) == 0:
            raise ValueError("DeepSeek API返回无效响应")

        return json_data["choices"][0]["message"]["content"]


class DeepSeek(BaseModel):
    def __init__(self, **kwargs):
        self.deepseek = DeepSeekChat(
            DEEPSEEK_API_KEY=kwargs.get("DEEPSEEK_API_KEY"),
            type_model=kwargs.get("type_model", "deepseek-chat")
        )

    def think(self, messages):
        if isinstance(messages, str):
            return self.deepseek.chat(messages)
        if len(messages) == 0:
            return ''
        return self.deepseek.chat(PromptGenerate(messages))