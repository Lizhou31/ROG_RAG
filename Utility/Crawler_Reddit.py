import time
import asyncio
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
        self.driver = None
        self.wait = None
        self._setup_driver()

    def _setup_driver(self):
        """Setup the webdriver instance"""
        options = webdriver.ChromeOptions()
        #options.add_argument('--headless')
        self.driver = webdriver.Chrome(options=options)
        self.wait = WebDriverWait(self.driver, 10)

    def collect_post_urls(self, num_posts=10):
        """Collect post URLs from the search page"""
        try:
            self.driver.get(self.search_url)
            time.sleep(5)
            
            posts = self.wait.until(
                EC.presence_of_all_elements_located((By.XPATH, '//a[contains(@class, "absolute inset-0") and contains(@data-testid, "post-title")]'))
            )
            
            post_data = []
            for post in posts[:num_posts]:
                title = post.get_attribute('aria-label')
                url = post.get_attribute('href')
                if url and title:
                    post_data.append((title, url))
                    
            return post_data
                
        except Exception as e:
            print(f"Error collecting URLs: {e}")
            return []

    async def process_single_post(self, post_data, tab_id):
        """Process a single post in a new tab"""
        title, url = post_data
        
        try:
            # Execute JavaScript to open and switch to a new tab
            self.driver.execute_script(f'window.open("{url}", "tab{tab_id}");')
            self.driver.switch_to.window(f"tab{tab_id}")
            
            # Wait for content to load
            await asyncio.sleep(3)
            
            content_elements = self.wait.until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[class="text-neutral-content"] div.md.text-14 p'))
            )
            content = ' '.join([element.text for element in content_elements])
            
            # Close the tab and switch back to main window
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            
            return {
                "title": title,
                "url": url,
                "content": content
            }
            
        except Exception as e:
            print(f"Error processing post {url}: {e}")
            # Ensure we switch back to main window even if there's an error
            if len(self.driver.window_handles) > 1:
                self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            return {
                "title": title,
                "url": url,
                "content": ""
            }

    async def get_latest_posts(self, num_posts=10):
        """Async method to get latest posts"""
        post_data = self.collect_post_urls(num_posts)
        
        # Process posts concurrently using asyncio.gather
        tasks = []
        for i, data in enumerate(post_data):
            task = asyncio.create_task(self.process_single_post(data, f"tab{i}"))
            tasks.append(task)
        
        # As each task completes, yield the result
        for task in asyncio.as_completed(tasks):
            result = await task
            yield result

    def close(self):
        """Cleanup resources"""
        if self.driver:
            self.driver.quit()

async def main():
    # Example usage
    crawler = RedditCrawler("mouse")
    try:
        async for post in crawler.get_latest_posts(3):
            print(f"Title: {post['title']}")
            print(f"URL: {post['url']}")
            print(f"Content: {post['content'][:100]}...")
            print("-" * 50)
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
