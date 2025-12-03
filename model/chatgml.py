from langchain.schema import HumanMessage, SystemMessage, AIMessage
from transformers import AutoTokenizer, AutoModel
from model.base.BaseModel import BaseModel
from langchain_openai import ChatOpenAI
from util import PromptGenerate

class chatGML():
    def __init__(self,model_path,quantizeV = 4):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path,trust_remote_code=True).quantize(quantizeV).cuda()
        self.model = self.model.eval()
        self.history = []

    def chat(self,prompt):
        text, self.history = self.model.chat(self.tokenizer, prompt, history=[])
        return text

class GML(BaseModel):
    def __init__(self,**kwargs):
        self.chatgml = chatGML(kwargs.get("modelpath"),kwargs.get("quantizeV"))
        self.llm = ChatOpenAI(openai_api_key='sk-xxx')

    def think(self, messages):
        if isinstance(messages, str):
            return self.chatgml.chat(messages)
        if len(messages) == 0:
            return ''
        return self.chatgml.chat(PromptGenerate(messages))


