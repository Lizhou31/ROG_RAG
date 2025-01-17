from Utility.Crawler_Forum import Forum_Crawler
from Utility.Crawler_Reddit import RedditCrawler
from Utility.ROG_RAG import RAGRetriever, VectorStoreManager, DocumentPreparation, OpenAILLM, OllamaLLM

def main():
    # Document preparation phase with Ollama
    doc_prep = DocumentPreparation(llm_provider=OpenAILLM(model_name="gpt-4o-mini"))
    documents = doc_prep.load_documents()
    documents = doc_prep.add_metadata(documents)
    
    # Vector store creation/loading
    vector_manager = VectorStoreManager()
    vector_store = vector_manager.create_or_load_vector_store(documents)
    
    # Create forum crawler instance
    forum_crawler = Forum_Crawler()
    
    # Get latest 10 posts from forum
    forum_posts = forum_crawler.get_latest_posts(10)
    
    # Print forum posts
    print("Forum Posts:")
    forum_crawler.print_posts(forum_posts)
    
    # Create Reddit crawler instance with search keyword
    reddit_crawler = RedditCrawler("mouse")
    
    # Get latest 10 posts from Reddit
    reddit_posts = reddit_crawler.get_latest_posts(10)
    
    # Print Reddit posts
    print("Reddit Posts:")
    reddit_crawler.print_posts(reddit_posts)
    
    # Create RAG retriever instance with Ollama
    retriever = RAGRetriever(vector_store, llm_provider=OllamaLLM(model_name="llama3.2"))
    
    # Process forum posts with RAG and save to markdown
    with open('result.md', 'w') as f:
        f.write("# RAG Results for Forum Posts\n\n")
        for topic, url, content in forum_posts:
            result = retriever.get_result(topic, content)
            f.write(f"## {topic}\n")
            f.write(f"**Content:** {content}\n\n")
            f.write(f"**URL:** {url}\n\n")
            f.write(f"**RAG Result:** {result}\n\n")
        
        f.write("# RAG Results for Reddit Posts\n\n")
        for title, url, content in reddit_posts:
            result = retriever.get_result(title, content)
            f.write(f"## {title}\n")
            f.write(f"**Content:** {content}\n\n") 
            f.write(f"**URL:** {url}\n\n")
            f.write(f"**RAG Result:** {result}\n\n")
    
    print("Results saved to result.md")

if __name__ == "__main__":
    main()
