import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from abc import ABC, abstractmethod

class PostFilter(ABC):
    @abstractmethod
    def should_skip(self, post) -> bool:
        pass

class RedditCrawler:
    def __init__(self, search_keyword):
        self.search_url = f'https://www.reddit.com/r/ASUSROG/search/?q={search_keyword}&sort=new'
        self.filters = []
        # Add default filters
        
        # Initialize webdriver with headless mode
        options = webdriver.ChromeOptions()
        #options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def add_filter(self, post_filter: PostFilter):
        """Add a new filter to the crawler"""
        self.filters.append(post_filter)

    def should_skip_post(self, post) -> bool:
        """Check if post should be skipped based on all filters"""
        return any(f.should_skip(post) for f in self.filters)

    def extract_post_data(self, post):
        """Extract title and content from a Reddit post"""
        try:
            title = post.find_element(By.TAG_NAME, 'h3').text.strip()
        except:
            title = ''
            
        try:
            content = post.find_element(By.CLASS_NAME, '_292iotee39Lmt0MkQZ2hPV').text.strip()
        except:
            content = ''
            
        return title, content

    def get_latest_posts(self, num_posts=10):
        latest_posts = []
        
        try:
            # Navigate directly to search URL
            self.driver.get(self.search_url)
            
            # Wait for search results to load
            time.sleep(5)
            
            # Find all posts with a more reliable selector
            posts = self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@class, "absolute inset-0") and contains(@data-testid, "post-title")]'))
            )

            for post in posts:
                if len(latest_posts) >= num_posts:
                    break

                if self.should_skip_post(post):
                    continue

                title = post.get_attribute('aria-label')
                url = post.get_attribute('href')
                
                # Click on post to get content
                post.click()
                time.sleep(5)
                
                try:
                    content_elements = self.wait.until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[class="text-neutral-content"] div.md.text-14 p'))
                    )
                    content = ' '.join([element.text for element in content_elements])
                except:
                    content = ''
                    
                latest_posts.append((title, url, content))
                
                # Go back to search results
                self.driver.back()
                time.sleep(2)

        except Exception as e:
            print(f"Error fetching posts: {e}")
        
        finally:
            self.driver.quit()

        return latest_posts

    def print_posts(self, posts):
        if posts:
            for i, (title, url, content) in enumerate(posts, start=1):
                print(f"Post {i} Title: {title}")
                print(f"Content: {content}")
                print(f"Url: {url}\n")
        else:
            print("Failed to retrieve the latest posts.")

def main():
    # Create crawler instance with search keyword
    crawler = RedditCrawler("mouse")
    
    # Get latest 10 posts
    latest_posts = crawler.get_latest_posts(1)
    
    # Print the posts
    crawler.print_posts(latest_posts)

if __name__ == "__main__":
    main()
