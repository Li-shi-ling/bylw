from langchain.schema import HumanMessage, SystemMessage, AIMessage
from model.base.BaseModel import BaseModel
from langchain_openai import ChatOpenAI
import http.client
import logging
import json

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

class chatGPT():
    def __init__(self,OPENAI_API_KEY,type_model,url):
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.type_model = type_model
        self.url = url
        self.headers = {'Authorization': 'Bearer ' + self.OPENAI_API_KEY,'User-Agent': 'Apifox/1.0.0 (https://apifox.com)', 'Content-Type': 'application/json'}
        self.modellist = self.Getmodellist()
        if not self.type_model in self.modellist:
            for name in self.modellist:
                if "gpt" in name:
                    self.type_model = name
                    break
            else:
                logging.info("该api不支持gpt")

    def chat(self, inputtext):
        for i in range(5):
            try:
                out_text = self.get_ai(inputtext)
            except Exception as e:
                logging.info(f"出现错误:{e}")
                continue
            break
        else:
            out_text = "出现错误"
        return out_text

    def get_ai(self,text):
        logging.info(f"text:{text}")
        conn = http.client.HTTPSConnection(self.url)
        payload = json.dumps({
            "model": self.type_model,
            "messages": [{'role': 'user', 'content': text}]
        })
        conn.request("POST", "/v1/chat/completions", payload, self.headers)
        res = conn.getresponse()
        data = res.read().decode("utf-8")
        json_data = json.loads(data)
        return f"{json_data['choices'][0]['message']['content']}"

    def Getmodellist(self):
        conn = http.client.HTTPSConnection(self.url)
        payload = ''
        headers = {
            'Authorization': 'Bearer ' + self.OPENAI_API_KEY,
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
        }
        conn.request("GET", "/v1/models", payload, headers)
        res = conn.getresponse()
        modellist = []
        for data in json.loads(res.read().decode("utf-8"))["data"]:
            modellist.append(data["id"])
        return modellist

class GPT(BaseModel):
    def __init__(self,**kwargs):
        self.chatgpt = chatGPT(OPENAI_API_KEY=kwargs.get("OPENAI_API_KEY"),type_model=kwargs.get("type_model","gpt-3.5-turbo"),url=kwargs.get("url"))
        self.llm = ChatOpenAI(openai_api_key='sk-xxx')

    def think(self, messages):
        if isinstance(messages, str):
            return self.chatgpt.chat(messages)
        if len(messages) == 0:
            return ''
        return self.chatgpt.chat(PromptGenerate(messages))


