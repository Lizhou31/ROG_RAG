from langchain.prompts import PromptTemplate
try:
    from LLM_Provider import BaseLLM, OpenAILLM, OllamaLLM
    from Vector_DB import VectorStoreManager
    from Doc_gradder import DocGrader
except ImportError:
    from Utility.LLM_Provider import BaseLLM, OpenAILLM, OllamaLLM
    from Utility.Vector_DB import VectorStoreManager
    from Utility.Doc_gradder import DocGrader

# if need to prepare the documents
from langchain.schema import Document
try:
    from Utility.Data_Preparation import DocumentPreparation
except ImportError:
    from Data_Preparation import DocumentPreparation

class RAGRetriever:
    def __init__(self, vector_store, llm_provider: BaseLLM = None):
        self.llm_provider = llm_provider if llm_provider else OpenAILLM()
        self.llm = self.llm_provider.get_llm()
        self.vector_store = vector_store
        self.prompt = self._create_prompt()
        self.retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

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
        
        If you don't get any retrieved docs, just return "Unknown" and nothing else.

        Please only output in json format but no other text. The summary should be a short summary of the forum post.
        "products":[product1 name, product2 name, product3 name...], "summary":[summary]
        """
        return PromptTemplate.from_template(template)

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_doc_metadata(self, docs):
        return "\n\n".join(f"Product: {doc.metadata['Product Name']}\nDescription: {doc.metadata['short_description']}" for doc in docs)

    def get_result(self, topic, description):
        query = f"{topic} {description}"
        retrieved_docs = self.retriever.invoke(query)
        doc_metadata = self.get_doc_metadata(retrieved_docs)
        doc_grader = DocGrader()
        relevance_score = doc_grader.grade_retrieval(topic, description, doc_metadata)
        # Filter out docs with low relevance scores
        filtered_docs = [doc for i, doc in enumerate(retrieved_docs) 
                        if relevance_score["relevance"][i] >= 6]
        
        # If no docs meet relevance threshold, return empty input
        if not filtered_docs:
            input_data = {
                "topic": topic,
                "description": description,
                "retrieved_docs": ""
            }
        else:
            input_data = {
                "topic": topic,
                "description": description,
                "retrieved_docs": self.format_docs(filtered_docs)
            }
        formatted_input = self.prompt.format(**input_data)
        result = self.llm.invoke(formatted_input)
        return self.llm_provider.process_result(result)
    
def main():
    # Document preparation phase with gpt-4o-mini
    # doc_prep = DocumentPreparation(llm_provider=OpenAILLM(model_name="gpt-4o-mini"))
    # documents = doc_prep.load_documents()
    # documents = doc_prep.add_metadata(documents)
    # split_docs = doc_prep.split_documents(documents)
    # split_docs_with_description = doc_prep.add_description_metadata(split_docs)
    
    # Vector store creation/loading
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.create_or_load_vector_store()
    
    # Retrieval phase with different LLM if desired
    # OpenAI with gpt-4o-mini
    # retriever = RAGRetriever(vector_store, llm_provider=OpenAILLM(model_name="gpt-4o-mini"))
    # Ollama with llama3.2
    retriever = RAGRetriever(vector_store, llm_provider=OllamaLLM(model_name="llama3.2"))
    
    # topic = "ROG keris wireless aimpoint macros have a significant delay"
    # description = "I recently bought a rog keris wireless aimpoint and there are some games that i like to record keyboard macros for on my mouse side buttons, but when i am in game there is a significant delay from the point i press the button and when the recorded ac..."
    topic = "My Logitech G502 Hero is not working"
    description = "I recently bought a Logitech G502 Hero and it is not working. I have tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working. I have also tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working. I have also tried to reset the mouse and the receiver but it is still not working. I have also tried to update the driver but it is still not working."
    result = retriever.get_result(topic, description)
    print(result)
    print(result["products"])
    print(result["summary"])
    
if __name__ == "__main__":
    main()
