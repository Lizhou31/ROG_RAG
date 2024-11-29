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
                
        return documents

    def split_documents(self, documents):
        split_docs = []
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

    def add_metadata(self, docs, schema=None):
        """Add metadata to documents using the provided schema or default schema."""
        if schema is None:
            schema = self._get_default_schema()
        metadata_tagger = create_metadata_tagger(metadata_schema=schema, llm=self.llm)
        return metadata_tagger.transform_documents(docs)
        
    def _get_default_schema(self):
        """Default schema for ROG product metadata."""
        return {
            "properties": {
                "Product Name": {"type": "string"},
                "Product connection method": {"type": "string"},
            },
            "required": ["Product Name", "Product connection method"],
        }
        
    def add_product_metadata(self, docs):
        """Add basic product metadata."""
        return self.add_metadata(docs, self._get_default_schema())
        
    def add_custom_metadata(self, docs, custom_schema):
        """Add metadata using a custom schema."""
        return self.add_metadata(docs, custom_schema)

def main():
    # Initialize document preparation with OpenAI LLM
    doc_prep = DocumentPreparation(llm_provider=OpenAILLM(model_name="gpt-4o-mini"))
    
    # Load and process documents
    print("Loading and splitting documents...")
    documents = doc_prep.load_documents()
    
    print("Adding metadata to documents...")
    documents_with_metadata = doc_prep.add_product_metadata(documents)
    
    print("Splitting documents...")
    split_docs = doc_prep.split_documents(documents_with_metadata)
    
    
    print(f"Processed {len(split_docs)} documents")
    
    # Print sample of processed documents
    if split_docs:
        print("\nSample document:")
        sample_doc = split_docs[0]
        print(f"Content: {sample_doc.page_content[:200]}...")
        print(f"Metadata: {sample_doc.metadata}")

if __name__ == "__main__":
    main()

