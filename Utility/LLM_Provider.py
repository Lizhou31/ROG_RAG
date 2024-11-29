import json
from abc import ABC, abstractmethod
from langchain_community.llms import Ollama
from langchain_openai import ChatOpenAI

class BaseLLM(ABC):
    @abstractmethod
    def get_llm(self):
        pass
    
    @abstractmethod
    def process_result(self, result):
        pass

class OllamaLLM(BaseLLM):
    def __init__(self, model_name="llama3.2", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
        
    def get_llm(self):
        return Ollama(model=self.model_name, temperature=self.temperature)
    
    def process_result(self, result):
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"products": [], "summary": result}

class OpenAILLM(BaseLLM):
    def __init__(self, model_name="gpt-4o-mini", temperature=0):
        self.model_name = model_name
        self.temperature = temperature
        
    def get_llm(self):
        return ChatOpenAI(model=self.model_name, temperature=self.temperature)
    
    def process_result(self, result):
        try:
            return json.loads(result.content)
        except json.JSONDecodeError:
            return {"products": [], "summary": result.content}
