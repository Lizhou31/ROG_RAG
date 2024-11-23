import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from Utility.Crawler_Forum import Forum_Crawler
from Utility.Crawler_Reddit import RedditCrawler
from Utility.ROG_RAG import RAGRetriever, VectorStoreManager, DocumentPreparation, OpenAILLM, OllamaLLM

app = Flask(__name__)

# Initialize vector store once at startup
doc_prep = DocumentPreparation(llm_provider=OpenAILLM(model_name="gpt-4o-mini"))
vector_manager = VectorStoreManager()
vector_store = vector_manager.create_or_load_vector_store()

# Create RAG retriever instance once
retriever = RAGRetriever(vector_store, llm_provider=OllamaLLM(model_name="llama3.2"))

@app.route('/')
def index():
    return render_template('index.html', forum_results=[], reddit_results=[])

@app.route('/get_forum_results')
def get_forum_results():
    # Create forum crawler instance
    forum_crawler = Forum_Crawler()
    
    # Get latest 5 posts from forum
    forum_posts = forum_crawler.get_latest_posts(5)
    
    # Process posts with RAG
    forum_results = [
        {
            "topic": topic,
            "url": url,
            "content": content,
            "rag_result": retriever.get_result(topic, content)
        }
        for topic, url, content in forum_posts
    ]
    
    return jsonify(forum_results)

@app.route('/get_reddit_results') 
def get_reddit_results():
    # Create Reddit crawler instance with search keyword
    reddit_crawler = RedditCrawler("mouse")
    
    # Get latest 5 posts from Reddit
    reddit_posts = reddit_crawler.get_latest_posts(5)
    
    # Process posts with RAG
    reddit_results = [
        {
            "title": title,
            "url": url,
            "content": content,
            "rag_result": retriever.get_result(title, content)
        }
        for title, url, content in reddit_posts
    ]
    
    return jsonify(reddit_results)

if __name__ == "__main__":
    app.run(debug=True)