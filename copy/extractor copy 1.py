import requests
from typing import List, Literal
from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urljoin
        
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


def page_content_extractor(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    extracted_text = []
    skip_next = False  # To skip duplicate words
    
    # parsed_url = urlparse("https://wordpress.org/data-liberation/guides/")
    # print("x................", parsed_url, parsed_url.scheme, parsed_url.netloc)
    # if not parsed_url.scheme or not parsed_url.netloc:
    #     return "Invalid"

    # base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
    
    

    for _, element in enumerate(soup.descendants):
        if skip_next:
            skip_next = False
            continue

        if element.name == "a" and element.get("href"):  # Process hyperlinks
            text = element.get_text(strip=True) or "Link"
            extracted_text.append(f"{text} (Link: {element['href']})")
            skip_next = True  # Skip the duplicate plain text after link

        elif element.name == "img" and element.get("src"):  # Process image links
            alt_text = element.get("alt", "No description")
            extracted_text.append(f"(Image Link: {element['src']} - Alt: {alt_text})")

        elif isinstance(element, str) and element.strip():  # Normal visible text
            extracted_text.append(element.strip())

    return " ".join(extracted_text)

# Example HTML
html_content = """
<html>
  <body>
    <h1>Welcome</h1>
    <p>This is a <a href="https://example.com/blog">blog post</a>.</p>
    <p>Check this image: <img src="https://example.com/image.jpg" alt="A nice image"></p>
  </body>
</html>
"""

# Extract formatted text
# formatted_text = page_content_extractor(html_content)
# print(formatted_text)


class ContentExtractor:
    
    def __init__(self, vector_store:Chroma, url:str|None) -> None:
        self.vector_store = vector_store
        self.extracted_links:List[str] = []
        self.extractor_tool: Literal['selenium', 'bs4'] = 'bs4'
            
    def extract_links(self, url:str):
        extracted_links = []
        
        if self.is_sitemap(url):
            extracted_links = self.fetch_sitemap_links(url)
        else:
            url_type = self.verify_url_type(url)
            if url_type == "Normal":
                extracted_links = self.extract_website_links(url)
            elif url_type == "Wordpress":
                extracted_links = self.extract_wordpress_links(url)
                
        self.extracted_links = extracted_links      
        return extracted_links
      
    def is_sitemap(self, url):
        """Check if a given URL is a sitemap based on common patterns."""
        return any(keyword in url.lower() for keyword in ["sitemap.xml", "sitemap_index.xml", "sitemap"])

    def fetch_sitemap_links(self, sitemap_url):
        """Fetch all URLs from a sitemap."""
        try:
            response = requests.get(sitemap_url, timeout=10)
            response.raise_for_status()
            urls = []

            if response.headers['Content-Type'].startswith("application/xml") or "xml" in response.text[:100]:
                root = ET.fromstring(response.text)
                for elem in root.iter():
                    if 'loc' in elem.tag:
                        urls.append(elem.text.strip())
            return urls

        except requests.RequestException as e:
            print(f"Error fetching sitemap: {e}")
            return []

    def verify_url_type(self, url:str) -> Literal['Invalid', 'Wordpress', 'Normal']:
        try:
            # Parse URL to check for validity
            parsed_url = urlparse(url)
            if not parsed_url.scheme or not parsed_url.netloc:
                return "Invalid"

            # Send request to the main URL to check if it's accessible
            response = requests.get(url, timeout=5)
            if response.status_code != 200:
                return "Invalid"
                
            headers = response.headers
            if ((("X-Powered-By" in headers) and ("WordPress" in headers["X-Powered-By"]))
                or (("X-Generator" in headers) and ("WordPress" in headers["X-Generator"]))):
                return "Wordpress"

            return "Normal"
        except requests.exceptions.RequestException:
            return "Invalid"
      
    def extract_website_links(self, url:str):
        """Extract all internal page URLs from a given webpage."""
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            
            # Check if the response is successful
            if response.status_code != 200:
                raise Exception(f"Failed to fetch the page, status code: {response.status_code}")
            
            # Check if content is blocked
            if self._is_page_blocked(response.text):
                raise Exception("Page is blocked by CAPTCHA or login requirements.")
            
            soup = BeautifulSoup(response.text, "html.parser")

            domain = urlparse(url).netloc
            urls = set()

            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == domain:  # Ensure it's the same domain
                    urls.add(full_url)

            return list(urls)

        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return []
    
    def extract_wordpress_links(self, url:str):
        try:
            # Parse and construct base URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
            # page_slug = url.strip('/').split('/')[-1]

            #? Currently we are fetching maximum 100 pages, but we can make it dynamic in future update
            url_suffix = 'wp-json/wp/v2/pages?_fields=link&per_page=100'
            wp_api_url = urljoin(base_url, url_suffix)
            response = requests.get(wp_api_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

            if response.status_code != 200:
                return []  # Not a WordPress site or API is disabled

            # Extract page URLs from JSON response
            pages_data = response.json()
            page_links = [page['link'] for page in pages_data if 'link' in page]

            return page_links

        except requests.exceptions.RequestException:
            return []
    
    def extract_content(self, urls:List[str])->dict[str, List[str]]:
        content = {}
        
        for url in urls:
            url_type = self.verify_url_type(url)
            
            if url_type == "Normal":
                content[url] = self.extract_website_content(url)
            elif url_type == 'Wordpress':
                content[url] = self.extract_wordpress_content(url)
            else:
                content[url] = None

        return content
    
    def extract_website_content(self, url:str):
        print(f"Trying to extract content from: {url}")

        if self.extractor_tool == 'bs4':
            # Try extracting with BeautifulSoup first
            if content := self.extract_with_bs4(url):
                print("Extracted using BeautifulSoup")
                return content
        else:
            print("Falling back to Selenium...")
            if content := self.extract_with_selenium(url):
                print("Extracted using Selenium")
                self.extractor_tool = 'selenium'       
                return content

        # If both methods fail, return an error
        print("Failed to extract content using both BeautifulSoup and Selenium.")
        self.extractor_tool = 'bs4'       
        return None

    def extract_with_bs4(self, url, extract:Literal['link', 'content'] = 'content'):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            
            # Check if the response is successful
            if response.status_code != 200:
                raise Exception(f"Failed to fetch the page, status code: {response.status_code}")
            
            # Check if content is blocked
            if self._is_page_blocked(response.text):
                raise Exception("Page is blocked by CAPTCHA or login requirements.")
            
            return self.clean_content(response.text)
        
        except Exception as e:
            print(f"BS4 Extraction Failed: {e}")
            return None

    def extract_with_selenium(self, url, extract:Literal['link', 'content'] = 'content'):
        try:
            driver = self._get_selenium_driver() 
            driver.get(url)
            
            # Wait for page to fully load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
            except Exception:
                print("Warning: Page load timeout, continuing with available content.")
            
            # Extract page source
            page_source = driver.page_source
            driver.quit()
            
            if self._is_page_blocked(page_source):
                raise Exception("Page is blocked by CAPTCHA or login requirements.")
            
            return self.clean_content(page_source)
        
        except Exception as e:
            print(f"Selenium Extraction Failed: {e}")
            return None


    def extract_wordpress_content(self, url:str):
        parsed_url = urlparse(url)
        print("x................", parsed_url, parsed_url.scheme, parsed_url.netloc)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        page_slug = url.strip('/').split('/')[-1]
        
        url_suffix = f"wp-json/wp/v2/pages?slug={page_slug}&_fields=slug,link,content"
        page_api_url = urljoin(base_url, url_suffix)
    
    def clean_content(self, html_content):
        soup = BeautifulSoup(html_content, "html.parser")
        extracted_text = []
        skip_next = False  # To skip duplicate words
 
        for _, element in enumerate(soup.descendants):
            if skip_next:
                skip_next = False
                continue

            if element.name == "a" and element.get("href"):  # Process hyperlinks
                text = element.get_text(strip=True) or "Link"
                extracted_text.append(f"{text} (Link: {element['href']})")
                skip_next = True  # Skip the duplicate plain text after link

            elif element.name == "img" and element.get("src"):  # Process image links
                alt_text = element.get("alt", "No description")
                extracted_text.append(f"(Image Link: {element['src']} - Alt: {alt_text})")

            elif isinstance(element, str) and element.strip():  # Normal visible text
                extracted_text.append(element.strip())

        return " ".join(extracted_text)


        
    def _get_soup_instance(self, url:str):
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        
        # Check if the response is successful
        if response.status_code != 200:
            raise Exception(f"Failed to fetch the page, status code: {response.status_code}")
        
        # Check if content is blocked
        if self._is_page_blocked(response.text):
            raise Exception("Page is blocked by CAPTCHA or login requirements.")
        
        soup = BeautifulSoup(response.text, "html.parser")
        return soup
        
    def _get_selenium_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Set up the service for ChromeDriver
        service = Service(ChromeDriverManager().install())
        # Use the service and options together
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def _is_page_blocked(self, content):
        block_keywords = ["captcha", "verify", "verification", "robot", "login", "restricted access"]
        soup = BeautifulSoup(content, "html.parser")
        text_content = soup.get_text().lower()
        
        return any(keyword in text_content for keyword in block_keywords)
        





