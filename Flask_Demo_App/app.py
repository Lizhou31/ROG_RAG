import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify
from Utility.Crawler_Forum import Forum_Crawler
from Utility.Crawler_Reddit import RedditCrawler
from Utility.ROG_RAG import RAGRetriever

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', forum_results=[], reddit_results=[])

@app.route('/get_forum_results')
def get_forum_results():
    # Create forum crawler instance
    forum_crawler = Forum_Crawler()
    
    # Get latest 10 posts from forum
    forum_posts = forum_crawler.get_latest_posts(10)
    
    # Create RAG retriever instance
    retriever = RAGRetriever()
    
    # Process posts with RAG
    forum_results = [
        {
            "topic": topic,
            "url": url,
            "content": content,
            "rag_result": retriever.get_result(topic, content).content
        }
        for topic, url, content in forum_posts
    ]
    
    return jsonify(forum_results)

@app.route('/get_reddit_results') 
def get_reddit_results():
    # Create Reddit crawler instance with search keyword
    reddit_crawler = RedditCrawler("mouse")
    
    # Get latest 10 posts from Reddit
    reddit_posts = reddit_crawler.get_latest_posts(5)
    
    # Create RAG retriever instance
    retriever = RAGRetriever()
    
    # Process posts with RAG
    reddit_results = [
        {
            "title": title,
            "url": url,
            "content": content,
            "rag_result": retriever.get_result(title, content).content
        }
        for title, url, content in reddit_posts
    ]
    
    return jsonify(reddit_results)

if __name__ == "__main__":
    app.run(debug=True)