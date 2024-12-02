import os
from abc import ABC, abstractmethod
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class BaseVectorStore(ABC):
    @abstractmethod
    def create_vector_store(self, documents, embeddings):
        pass
    
    @abstractmethod 
    def load_vector_store(self, path, embeddings):
        pass

class FAISSVectorStore(BaseVectorStore):
    def create_vector_store(self, documents, embeddings):
        return FAISS.from_documents(documents, embeddings)
        
    def load_vector_store(self, path, embeddings):
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

class VectorStoreManager:
    def __init__(self, vector_store_path="VectorStore/vector_store_3.json", 
                 vector_store=None, embeddings=None):
        self.vector_store_path = vector_store_path
        self.vector_store = vector_store if vector_store else FAISSVectorStore()
        self.embeddings = embeddings if embeddings else OpenAIEmbeddings()
        
    def create_or_load_vector_store(self, documents=None):
        if not os.path.exists(self.vector_store_path) and documents:
            vector_store = self.vector_store.create_vector_store(documents, self.embeddings)
            vector_store.save_local(self.vector_store_path)
        else:
            vector_store = self.vector_store.load_vector_store(self.vector_store_path, self.embeddings)
        return vector_store