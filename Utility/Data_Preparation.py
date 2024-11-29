import glob
import json
from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveJsonSplitter
from LLM_Provider import BaseLLM, OpenAILLM

class DocumentPreparation:
    def __init__(self, directory_path='RAG_Files', llm_provider: BaseLLM = None):
        self.directory_path = directory_path
        self.llm_provider = llm_provider if llm_provider else OpenAILLM()
        self.json_splitter = RecursiveJsonSplitter(max_chunk_size=200)
        self.llm = self.llm_provider.get_llm()
        
    def load_documents(self):
        documents = []
        split_docs = []
        json_files = glob.glob(f"{self.directory_path}/Assets/*.json")
        
        for json_file in json_files:
            try:
                loader = JSONLoader(
                    file_path=json_file,
                    jq_schema='.',
                    text_content=False
                )
                documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
        
        for doc in documents:
            try:
                # Ensure content is proper JSON before splitting
                if isinstance(doc.page_content, str):
                    content = json.loads(doc.page_content)
                else:
                    content = doc.page_content
                
                # Convert content to a list if it's not already
                if not isinstance(content, list):
                    content = [content]
                    
                # Split each document and maintain metadata
                for item in content:
                    try:
                        splits = self.json_splitter.create_documents([item])
                        # Copy metadata from original document to splits
                        for split in splits:
                            split.metadata.update(doc.metadata)
                        split_docs.extend(splits)
                    except Exception as e:
                        print(f"Error splitting item in document: {e}")
                        continue
                        
            except Exception as e:
                print(f"Error processing document: {e}")
                continue
        
        return split_docs

    def add_metadata(self, docs):
        schema = {
            "properties": {
                "Product Name": {"type": "string"},
                "Product connection method": {"type": "string"},
            },
            "required": ["Product Name", "Product connection method"],
        }
        metadata_tagger = create_metadata_tagger(metadata_schema=schema, llm=self.llm)
        return metadata_tagger.transform_documents(docs)
