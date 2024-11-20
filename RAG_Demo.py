import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

# Prepare RAG
documents = []
with open('RAG_Files/Mouse.txt') as file:
    text = file.read()
    chunks = text.split("<chunk>")
    chunks = [chunk.replace("\n", "") for chunk in chunks if chunk]
    documents = [Document(page_content=chunk) for chunk in chunks]

# Embeddings and Vector Store
if not os.path.exists("vector_store.json"):
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    vector_store.save_local("vector_store.json")
else:
    # allow_dangerous_deserialization is used to load the vector store from the local file
    vector_store = FAISS.load_local("vector_store.json", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Prompt template
template = """
You are an helpful assistant that specializes in categorizing forum posts based on the database.
The database contains information about the ROG products.
1. Retrieve the most relevant products from the database base on the input topic and description.   
2. Categorize the input topic and description into one of the following categories:

Input: 
- Forum Topic: {topic}
- Forum Description: {description}

Retrieved Products description:
{retrieved_docs}

If you cannot find the relevant products from the database, just return "Unknown".

Output:
[Product 1, Product 2, Product 3...]
"""

prompt = PromptTemplate.from_template(template)

# Retrieve RAG
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

topic = "ROG keris wireless aimpoint macros have a significant delay"
description = "I recently bought a rog keris wireless aimpoint and there are some games that i like to record keyboard macros for on my mouse side buttons, but when i am in game there is a significant delay from the point i press the button and when the recorded ac..."

# Combine topic and description for retrieval
query = f"{topic} {description}"
retrieved_docs = retriever.invoke(query)
# Prepare the input for the LLM
input_data = {
    "topic": topic,
    "description": description,
    "retrieved_docs": format_docs(retrieved_docs)
}

# Use the prompt to format the input
formatted_input = prompt.format(**input_data)

# Invoke the LLM with the formatted input
result = llm.invoke(formatted_input)

print(result)