from langchain.schema import HumanMessage, SystemMessage
from model.base.BaseModel import BaseModel
from model.EmotionRecognition.base import EmotionBase

#基于llm的情绪识别
class Emotion(EmotionBase):
    def __init__(self, model: BaseModel):
        self.brain = model
        self.moods = ['表现自己可爱', '生气', '高兴兴奋', '难过', '平常聊天', '温柔', '尴尬害羞']
        self.role = f'''Analyzes the sentiment of a given text said by a girl. When it comes to intimate behavior, such as sexual activity, one should reply with a sense of shyness. Response with one of {self.moods}.'''

    def think(self, text: str):
        message = [
            SystemMessage(content=self.role),
            HumanMessage(content=f'''Response with one of {self.moods} for the following text:\n"{text}"''')
        ]
        reply = self.brain.think_nonstream(message)
        for mood in self.moods:
            if mood in reply:
                return mood
        return '平常聊天'
