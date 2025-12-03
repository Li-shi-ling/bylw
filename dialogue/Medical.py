from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from dialogue.base import DialogueBase
from util.tool import PromptGenerate
import time

class Medical(DialogueBase):
    def __init__(self,**kwargs):
        # 大型语言模型
        self.model = kwargs.get("model")
        # 血液诊断知识图谱
        self.KG_blood = kwargs.get("KG_blood")
        # 病例知识图谱
        self.KG_record = kwargs.get("KG_record")
        # 大脑图像知识图谱
        self.KG_brain = kwargs.get("KG_brain")

    def ask(self, data: dict) -> str:
        blood = data.get("blood", '')
        record = data.get("record", '')
        brain = data.get("brain", '')

        prompt = []

        if blood != '':
            blood_prompt = SystemMessage(content=str(self.KG_blood.query(blood)))
        else:
            blood_prompt = SystemMessage(content='')
        prompt.append(blood_prompt)

        if record != '':
            record_prompt = SystemMessage(content=str(self.KG_record.query(record)))
        else:
            record_prompt = SystemMessage(content='')
        prompt.append(record_prompt)

        if brain != '':
            brain_prompt = SystemMessage(content=str(self.KG_brain.query(brain)))
        else:
            brain_prompt = SystemMessage(content='')
        prompt.append(brain_prompt)

        rawdata = f"""
        Please perform an integrated medical analysis of this patient using the following tripartite dataset:
        
        **BLOOD TEST RESULTS:**
        {blood}
        
        **PATIENT MEDICAL RECORD:**
        {record}
        
        **BRAIN DIAGNOSTIC DATA:**
        {brain}
        
        Provide a comprehensive health assessment and clinical recommendations based on the correlation of findings across all three data sources.
        """

        prompt.append(SystemMessage(content=rawdata))

        reply = self.model.think(prompt)

        return reply
