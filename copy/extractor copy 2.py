import requests
from typing import List, Literal
from bs4 import BeautifulSoup, Comment
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

# Fetching links and content using bs4 - Done


class ContentExtractor:
    
    def __init__(self, vector_store=None) -> None:
        self.vector_store = vector_store
        self.extracted_links:List[str] = []
        self.extractor_tool: Literal['selenium', 'bs4'] = 'bs4'
          
          
    # Links extractor functions  
    def extract_links(self, url:str):
        extracted_links = []
        
        if self.__is_sitemap(url):
            extracted_links = self.fetch_sitemap_links(url)
        else:
            url_type = self.__verify_url_type(url)
            if url_type == "Normal":
                extracted_links = self.extract_website_links(url)
            elif url_type == "Wordpress":
                extracted_links = self.extract_wordpress_links(url)
                
        self.extracted_links = extracted_links      
        return extracted_links

    def fetch_sitemap_links(self, sitemap_url)-> List[str]:
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
    
    def extract_wordpress_links(self, url:str)-> List[str]:
        try:
            # Parse and construct base URL
            parsed_url = urlparse(url)
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"

            #? Currently we are fetching maximum 100 pages, but we can make it dynamic in future update
            url_suffix = 'wp-json/wp/v2/pages?_fields=link&per_page=100'
            wp_api_url = urljoin(base_url, url_suffix)
            response = requests.get(wp_api_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

            if response.status_code != 200:
                return []  

            # Extract page URLs from JSON response
            pages_data = response.json()
            page_links = [page['link'] for page in pages_data if 'link' in page]

            return page_links

        except requests.exceptions.RequestException:
            return []

    def extract_website_links(self, url:str)-> List[str]:
        print(f"Trying to extract link from: {url}")

        # Try extracting with BeautifulSoup first
        if links := self._extract_with_bs4(url, extract='links'):
            print("Extracted using BeautifulSoup")
            return links
        
        print("Falling back to Selenium...")
        if links := self._extract_with_selenium(url, extract='links'):
            print("Extracted using Selenium")
            return links

        # If both methods fail, return an error
        print("Failed to extract content using both BeautifulSoup and Selenium.")
        return []


    # Content extractor functions
    def extract_content(self, urls:List[str])->dict[str, List[str]]:
        content = {}
        
        for url in urls:
            url_type = self.__verify_url_type(url)
            
            if url_type == "Normal":
                content[url] = self.extract_website_content(url)
            elif url_type == 'Wordpress':
                content[url] = self.extract_wordpress_page_content(url)
            else:
                content[url] = None

        return content
        
    def extract_wordpress_page_content(self, url:str):
        parsed_url = urlparse(url)
        
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        page_slug = url.strip('/').split('/')[-1]
        
        url_suffix = f"wp-json/wp/v2/pages?slug={page_slug}&_fields=slug,link,content"
        page_api_url = urljoin(base_url, url_suffix)
        response = requests.get(page_api_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code != 200:
            return None

        # Extract page URLs from JSON response
        content = response.json()[0]
        page_content = None
        if 'content' in content:
            page_content = ''
            for field, value in content.items():
                if field == 'content':
                    value = self.__clean_content(value.get('rendered', ''))
                page_content += f"{field}:{value}\n"
        
        return page_content            
    
    def extract_website_content(self, url:str):
        print(f"Trying to extract content from: {url}")

        if self.extractor_tool == 'bs4':
            # Try extracting with BeautifulSoup first
            if content := self._extract_with_bs4(url, extract='content'):
                print("Extracted using BeautifulSoup")
                return content
            
            # Retrying if not fetched with bs4
            elif content := self._extract_with_selenium(url, extract='content'):
                print("Extracted using Selenium")
                self.extractor_tool = 'selenium'       
                return content
        else:
            print("Falling back to Selenium...")
            if content := self._extract_with_selenium(url, extract='content'):
                print("Extracted using Selenium")
                self.extractor_tool = 'selenium'       
                return content

        # If both methods fail, return an error
        print("Failed to extract content using both BeautifulSoup and Selenium.")
        self.extractor_tool = 'bs4'       
        return None


    # BS4 and selenium functions
    def _extract_with_bs4(self, url:str, extract:Literal['links', 'content'] = 'content'):
        try:
            response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)

            # Check if the response is successful
            if response.status_code != 200:
                raise Exception(f"Failed to fetch the page, status code: {response.status_code}")
            
            # Check if content is blocked
            soup = BeautifulSoup(response.text, "html.parser")
            if self.__is_page_blocked('bs4', soup):
                raise Exception("Page is blocked by CAPTCHA or login requirements.")
            
            if extract == 'content':
                return self.__extract_content_using_bs4(response.text)
            else:
                return self.__extract_links_using_bs4(url, response.text)
        
        except Exception as e:
            print(f"BS4 Extraction Failed: {e}")
            return None

    def __extract_content_using_bs4(self, response):
        return self.__clean_content(response) 
     
    def __extract_links_using_bs4(self, url, response):
        try:
            soup = BeautifulSoup(response, "html.parser")

            domain = urlparse(url).netloc
            urls = set()
            
            for link in soup.find_all("a", href=True):
                href = link["href"]
                full_url = urljoin(url, href)
                if urlparse(full_url).netloc == domain:  # Ensure it's the same domain
                    urls.add(full_url)

            return list(urls)
        except Exception as e:
            print(f"Error while fetching links using bs4: {str(e)}")
            return []
        
        
    def _extract_with_selenium(self, url, extract:Literal['links', 'content'] = 'content'):
        try:
            driver = self.__get_selenium_driver() 
            driver.get(url)
            
            # Wait for page to fully load
            try:
                WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            except Exception:
                print("Warning: Page load timeout, continuing with available content.")
            
            if extract == 'content':
                return self.__extract_content_using_selenium(driver)         
            else:
                return self.__extract_links_using_selenium(driver, url)
        except Exception as e:
            print(f"Selenium Extraction Failed: {e}")
            return None
     
    def __extract_content_using_selenium(self, driver):
        try:
            # Extract page source
            page_source = driver.page_source
            
            if self.__is_page_blocked('selenium', driver):
                raise Exception("Page is blocked by CAPTCHA or login requirements.")
                
            driver.quit()
            return self.__clean_content(page_source)
        except Exception as e:
            print(f"Error while fetching content using selenium : {str(e)}")
            return None
    
    def __extract_links_using_selenium(self, driver, url:str):
        try:
            domain = urlparse(url).netloc
            urls = set()

            # Find all links on the page
            link_elements = driver.find_elements(By.TAG_NAME, "a")
            for link in link_elements:
                if href := link.get_attribute("href"):
                    full_url = urljoin(url, href)
                    if urlparse(full_url).netloc == domain:  
                        urls.add(full_url)

            return list(urls)
        except Exception as e:
            print(f"Error while fetching links using selenium : {str(e)}")
            return []

      
    # Helper functions
    def __is_sitemap(self, url):
        """Check if a given URL is a sitemap based on common patterns."""
        return any(keyword in url.lower() for keyword in ["sitemap.xml", "sitemap_index.xml", "sitemap"])

    def __verify_url_type(self, url:str) -> Literal['Invalid', 'Wordpress', 'Normal']:
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
      
    def __clean_content(self, html_content):
        """
        Extracts only the visible content from HTML, removes scripts, styles, comments,
        and preserves useful elements like links and images.
        
        :param html_content: Raw HTML content.
        :return: Cleaned visible text with properly formatted links and images.
        """
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove <script>, <style>, and IE conditional comments
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Remove HTML comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        extracted_text = []
        skip_next = False  # To prevent duplicate text after links

        for element in soup.descendants:
            if skip_next:
                skip_next = False
                continue

            if element.name == "a" and element.get("href"):  # Extract hyperlinks
                text = element.get_text(strip=True) or "Link"
                extracted_text.append(f"{text} (Link: {element['href']})")
                skip_next = True  # Skip duplicate text after link

            elif element.name == "img" and element.get("src"):  # Extract images
                alt_text = element.get("alt", "No description")
                extracted_text.append(f"(Image: {element['src']} - Alt: {alt_text})")

            elif isinstance(element, str) and element.strip():  # Normal visible text
                extracted_text.append(element.strip())

        return " ".join(extracted_text)
        
    def __get_selenium_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # Set up the service for ChromeDriver
        service = Service(ChromeDriverManager().install())
        # Use the service and options together
        driver = webdriver.Chrome(service=service, options=chrome_options)
        return driver

    def __is_page_blocked(self, tool:Literal['bs4', 'selenium'] = 'bs4', tool_instance = None):
        
        if tool == 'selenium' and tool_instance:
            driver = tool_instance

            # Check for common CAPTCHA elements (iframe, div, img, input)
            captcha_keywords = ["captcha", "challenge", "verify", "recaptcha"]
            captcha_elements = driver.find_elements(By.XPATH, "//*[contains(@id, 'captcha') or contains(@class, 'captcha')]")
            if any(captcha_elements) or any(keyword in driver.page_source.lower() for keyword in captcha_keywords):
                return True

            # Check if a login and signup form is present
            login_elements = driver.find_elements(By.XPATH, "//form[contains(@action, 'login') or contains(@id, 'login') or contains(@class, 'login')]")
            signup_elements = driver.find_elements(By.XPATH, "//form[contains(@action, 'signup') or contains(@id, 'signup') or contains(@class, 'signup')]")
            if login_elements or signup_elements:
                return True

            # Check for "verification required" messages 
            # or if the page is blocked (Access Denied / Forbidden)
            verification_keywords = ["verify", "verification", "security check", "robot"]
            blocked_keywords = ["access denied", "forbidden", "blocked", "restricted", "please enable javascript"]
            keywords = verification_keywords + blocked_keywords
            if any(keyword in driver.page_source.lower() for keyword in keywords):
                return True
            
        elif tool == 'bs4' and tool_instance:
            soup = tool_instance

            # Check for CAPTCHA elements
            if soup.find("iframe", {"src": lambda x: x and "captcha" in x}):
                return True, "captcha"
            if soup.find("img", {"src": lambda x: x and "captcha" in x}):
                return True, "captcha"
            if soup.find("div", {"id": lambda x: x and "captcha" in x}):
                return True, "captcha"

            # Check for Login forms (input fields for username/password)
            if soup.find("form", {"id": lambda x: x and "login" in x}) or \
            soup.find("form", {"class": lambda x: x and "login" in x}):
                return True, "login"

            if soup.find("input", {"name": "username"}) and soup.find("input", {"type": "password"}):
                return True, "login"

            # Check for Signup forms
            if soup.find("form", {"id": lambda x: x and "signup" in x}) or \
            soup.find("form", {"class": lambda x: x and "signup" in x}):
                return True

            # Check for "Verification Required" messages
            verification_keywords = ["verify", "verification", "security check", "robot", "please prove"]
            blocked_keywords = ["access denied", "forbidden", "blocked", "restricted", "enable javascript"]
            keywords = verification_keywords + blocked_keywords
            if any(keyword in soup.get_text().lower() for keyword in keywords):
                return True
            
        return False
        
    def __is_needs_selenium(self, url:str):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        # If request fails or is blocked, use Selenium
        if response.status_code != 200:
            return True

        soup = BeautifulSoup(response.text, "html.parser")

        # Check if the page is empty or mostly scripts
        if not soup.find() or len(soup.find_all("script")) > len(soup.find_all()):
            return True

        # Check if the page has a message requiring JavaScript
        js_required_keywords = ["enable javascript", "please enable javascript", "requires javascript"]
        if any(keyword in response.text.lower() for keyword in js_required_keywords):
            return True
    
        return False
        

extractor = ContentExtractor()
# links = extractor.extract_links("http://books.toscrape.com/")
content = extractor.extract_content(["https://quotes.toscrape.com/js/"])
# print("x....................", links)
print("x....................", content)


