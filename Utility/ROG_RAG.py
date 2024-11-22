import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)

class RAGRetriever:
    def __init__(self, file_path='RAG_Files/Mouse.txt', vector_store_path="vector_store.json"):
        self.file_path = file_path
        self.vector_store_path = vector_store_path
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.documents = self._load_documents()
        self.documents = self._add_metadata(self.documents)
        self.vector_store = self._load_vector_store()
        self.prompt = self._create_prompt()
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def _load_documents(self):
        with open(self.file_path) as file:
            text = file.read()
            chunks = text.split("<chunk>")
            chunks = [chunk.replace("\n", "") for chunk in chunks if chunk]
            return [Document(page_content=chunk) for chunk in chunks]

    def _add_metadata(self, docs):
        schema = {
            "properties": {
                "Product Name": {"type": "string"},
                "Product connection method": {"type": "string"},
            },
            "required": ["Product Name", "Product connection method"],
        }
        metadata_tagger = create_metadata_tagger(metadata_schema=schema, llm=self.llm)
        return metadata_tagger.transform_documents(docs)
    
    def _load_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            embeddings = OpenAIEmbeddings()
            vector_store = FAISS.from_documents(self.documents, embeddings)
            vector_store.save_local(self.vector_store_path)
        else:
            vector_store = FAISS.load_local(self.vector_store_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        return vector_store

    def _create_prompt(self):
        template = """
        You are an helpful assistant that specializes in categorizing forum posts based on the database.
        The database contains information about the ROG products.
        1. Retrieve the most relevant products from the database base on the input topic and description.   
        2. Add a short summary about this post.

        Input: 
        - Forum Topic: {topic}
        - Forum Description: {description}

        All the result should be in English.
        Retrieved Products description:
        {retrieved_docs}

        If you cannot find the relevant products from the database, just return "Unknown" and nothing else.

        Output:
        [Product 1, Product 2, Product 3...]
        [Summary]
        """
        return PromptTemplate.from_template(template)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_result(self, topic, description):
        query = f"{topic} {description}"
        retrieved_docs = self.retriever.invoke(query)
        input_data = {
            "topic": topic,
            "description": description,
            "retrieved_docs": self.format_docs(retrieved_docs)
        }
        formatted_input = self.prompt.format(**input_data)
        result = self.llm.invoke(formatted_input)
        return result

def main():
    retriever = RAGRetriever()
    topic = "ROG keris wireless aimpoint macros have a significant delay"
    description = "I recently bought a rog keris wireless aimpoint and there are some games that i like to record keyboard macros for on my mouse side buttons, but when i am in game there is a significant delay from the point i press the button and when the recorded ac..."
    result = retriever.get_result(topic, description)
    print(result.content)

if __name__ == "__main__":
    main()

