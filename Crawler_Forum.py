import requests
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod

class PostFilter(ABC):
    @abstractmethod
    def should_skip(self, post) -> bool:
        pass

class PinnedPostFilter(PostFilter):
    def should_skip(self, post) -> bool:
        return bool(post.find('i', class_='custom-thread-floated'))

class ResolvedPostFilter(PostFilter):
    def should_skip(self, post) -> bool:
        return bool(post.find('i', class_='custom-thread-solved'))

class Forum_Crawler:
    def __init__(self, base_url='https://rog-forum.asus.com/t5/gaming-mice/bd-p/GGA_MS'):
        self.base_url = base_url
        self.filters = []
        # Add default filters
        self.add_filter(PinnedPostFilter())
        self.add_filter(ResolvedPostFilter())

    def add_filter(self, post_filter: PostFilter):
        """Add a new filter to the crawler"""
        self.filters.append(post_filter)

    def should_skip_post(self, post) -> bool:
        """Check if post should be skipped based on all filters"""
        return any(f.should_skip(post) for f in self.filters)

    def extract_post_data(self, post):
        """Extract topic and content from a post"""
        topic_link = post.find('a')
        topic = topic_link.text.strip() if topic_link else ''
        
        url = topic_link['href'] if topic_link else ''
        
        message_body = post.find('p')
        content = message_body.text.strip() if message_body else ''
        
        return topic, url, content

    def get_latest_posts(self, num_posts=10):
        latest_posts = []
        page_number = 1

        while len(latest_posts) < num_posts:
            url = f"{self.base_url}/page/{page_number}"
            response = requests.get(url)
            if response.status_code != 200:
                break

            soup = BeautifulSoup(response.content, 'html.parser')
            posts = soup.find_all('article', class_=lambda x: x and 'custom-message-tile' in x.split())

            for post in posts:
                if len(latest_posts) >= num_posts:
                    break

                if self.should_skip_post(post):
                    continue

                topic, url, content = self.extract_post_data(post)
                latest_posts.append((topic, url, content))

            page_number += 1

        return latest_posts

    def print_posts(self, posts):
        if posts:
            for i, (topic, url, content) in enumerate(posts, start=1):
                print(f"Post {i} Topic: {topic}")
                print(f"Content: {content}\n")
                print(f"Url: {url}\n")
        else:
            print("Failed to retrieve the latest posts.")

def main():
    # Create crawler instance
    crawler = Forum_Crawler()
    
    # Get latest 5 posts
    latest_posts = crawler.get_latest_posts(10)
    
    # Print the posts
    crawler.print_posts(latest_posts)

if __name__ == "__main__":
    main()
