import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, Response, stream_with_context
from Utility.Crawler_Forum import Forum_Crawler
from Utility.Crawler_Reddit import RedditCrawler
from Utility.ROG_RAG import RAGRetriever, VectorStoreManager, DocumentPreparation, OpenAILLM, OllamaLLM
import json
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import asyncio
from multiprocessing import Value
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
    def generate_forum_results():
        try:
            forum_crawler = Forum_Crawler()
            result_queue = queue.Queue()
            completed_count = Value('i', 0)
            
            async def process_post(index, post_data):
                try:
                    post_result = await forum_crawler.process_single_post(post_data)
                    
                    try:
                        rag_result = retriever.get_result(post_result["topic"], post_result["content"])
                        result = {
                            "index": index,
                            "topic": post_result["topic"],
                            "url": post_result["url"],
                            "content": post_result["content"],
                            "products": rag_result["products"],
                            "summary": rag_result["summary"]
                        }
                        result_queue.put(result)
                        with completed_count.get_lock():
                            completed_count.value += 1
                    except Exception as e:
                        result_queue.put({
                            "index": index,
                            "error": f"RAG processing error: {str(e)}"
                        })
                        with completed_count.get_lock():
                            completed_count.value += 1
                
                except Exception as e:
                    result_queue.put({
                        "index": index,
                        "error": str(e)
                    })
                    with completed_count.get_lock():
                        completed_count.value += 1

            def run_async_processing():
                async def process_all_posts():
                    posts = await forum_crawler.get_latest_posts(5)
                    tasks = []
                    for i, post in enumerate(posts):
                        task = asyncio.create_task(process_post(i, post))
                        tasks.append(task)
                    await asyncio.gather(*tasks)
                
                asyncio.run(process_all_posts())
            
            processing_thread = threading.Thread(target=run_async_processing)
            processing_thread.start()
            
            # Wait for initial posts to be collected
            while True:
                try:
                    result = result_queue.get(timeout=0.1)
                    yield f"data: {json.dumps(result)}\n\n"
                except queue.Empty:
                    if completed_count.value >= 5:  # Number of posts we requested
                        break
                    continue
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        finally:
            forum_crawler.close()

    return Response(
        stream_with_context(generate_forum_results()),
        mimetype='text/event-stream'
    )

@app.route('/get_reddit_results')
def get_reddit_results():
    def generate_reddit_results():
        try:
            crawler = RedditCrawler("mouse")
            post_data = crawler.collect_post_urls(5)
            
            result_queue = queue.Queue()
            completed_count = Value('i', 0)
            
            async def process_post(index, post_data):
                try:
                    title, url = post_data
                    post_result = await crawler.process_single_post((title, url), f"tab{index}")
                    
                    try:
                        rag_result = retriever.get_result(post_result["title"], post_result["content"])
                        result = {
                            "index": index,
                            "title": post_result["title"],
                            "url": post_result["url"],
                            "content": post_result["content"],
                            "products": rag_result["products"],
                            "summary": rag_result["summary"]
                        }
                        result_queue.put(result)
                        with completed_count.get_lock():
                            completed_count.value += 1
                    except Exception as e:
                        result_queue.put({
                            "index": index,
                            "error": f"RAG processing error: {str(e)}"
                        })
                        with completed_count.get_lock():
                            completed_count.value += 1
                
                except Exception as e:
                    result_queue.put({
                        "index": index,
                        "error": str(e)
                    })
                    with completed_count.get_lock():
                        completed_count.value += 1
            
            def run_async_processing():
                async def process_all_posts():
                    tasks = []
                    for i, post in enumerate(post_data):
                        task = asyncio.create_task(process_post(i, post))
                        tasks.append(task)
                    await asyncio.gather(*tasks)
                
                asyncio.run(process_all_posts())
            
            processing_thread = threading.Thread(target=run_async_processing)
            processing_thread.start()
            
            while completed_count.value < len(post_data):
                try:
                    result = result_queue.get(timeout=0.1)
                    yield f"data: {json.dumps(result)}\n\n"
                except queue.Empty:
                    continue
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
        finally:
            crawler.close()

    return Response(
        stream_with_context(generate_reddit_results()),
        mimetype='text/event-stream'
    )

if __name__ == "__main__":
    app.run(debug=True)