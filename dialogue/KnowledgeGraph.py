from dialogue.base import DialogueBase
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import HumanMessage,AIMessage,SystemMessage
from util.tool import PromptGenerate
import time

class KnowledgeGraph(DialogueBase):
    def __init__(self,**kwargs):
        self.model = kwargs.get("model")
        self.TextVectorization = kwargs.get("TextVectorization")
        self.VectorizedTextSimilarityMatching = kwargs.get("VectorizedTextSimilarityMatching")
        self.VectorizedTextSimilarityMatching.Setllm(self.model)
        self.VectorizedTextSimilarityMatching.SetGraphpath(kwargs.get("SetGraphpath"))
        self.top_n = kwargs.get("top_n")
        self.databases = kwargs.get("databases")
        self.charactor_prompt = ""
        self.chat_memory = ChatMessageHistory()
        self.history = ChatMessageHistory()
        self.summary = SystemMessage(content="")
        self.charactor_prompt = SystemMessage(content="")

    def ask(self, text: str) -> str:
        if text == '':
            return ''
        message = HumanMessage(content=text)
        if self.model.llm.get_num_tokens_from_messages([message]) + self.model.llm.get_num_tokens_from_messages(self.chat_memory.messages)>= 1536:
            self.summarize_memory()
        if self.model.llm.get_num_tokens_from_messages([message]) + self.model.llm.get_num_tokens_from_messages([self.summary])>= 1536:
            self.cut_summary()

        messages = [self.charactor_prompt,self.summary]

        #bgtime = time.time()
        relative_Knowledge = self.VectorizedTextSimilarityMatching.query(text,self.top_n,self.databases)
        #endtime = time.time()
        #print(f"time:{endtime - bgtime}")

        total_token,i,is_full = 0,0,False
        for i in range(len(relative_Knowledge)):
            if isinstance(relative_Knowledge[i],float):
                continue
            total_token += self.model.llm.get_num_tokens(relative_Knowledge[i])
            if (total_token >= 1024):
                is_full = True
        if is_full:
            relative_Knowledge = relative_Knowledge[:i]

        if len(relative_Knowledge) > 0:
            #print(f"SystemMessage:\n一共加载{len(relative_Knowledge)}条知识:\n" + ''.join([f'{i}.' + relative_Knowledge[i] + '\n' for i in range(len(relative_Knowledge))]))
            KnowledgePrompt = f'This following message is relative context for your response:\n\n{str(relative_Knowledge)}'
            memory_message = SystemMessage(content=KnowledgePrompt)
            messages.append(memory_message)

        self.chat_memory.messages.append(message)
        self.history.messages.append(message)
        messages.extend(self.chat_memory.messages)

        reply = self.model.think(messages)

        self.chat_memory.messages.append(AIMessage(content=reply))

        return reply

    def summarize_memory(self):
        prompt = PromptGenerate(self.chat_memory.messages)
        prompt_template = f"""Write a concise summary of the following, time information should be include:


        {prompt}


        CONCISE SUMMARY IN CHINESE LESS THAN 300 TOKENS:"""
        self.summary = SystemMessage(content=f"\n\nCHAT SUMMARY:{self.model.think_nonstream([SystemMessage(content=prompt_template)])}\n\n")
        self.chat_memory = ChatMessageHistory()

    def cut_summary(self):
        self.summary = SystemMessage(content="")
