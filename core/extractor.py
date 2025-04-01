import requests
import warnings
from typing import List, Literal
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup, Comment, XMLParsedAsHTMLWarning
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, urljoin
        
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# Able to find out is this a normal website url, a sitemap file url or wordpress website url
# Auto find out that scrapping needs bs4 or selenium - Done
# Auto track that if page is restricted or not - Done
# Fetching links and content using bs4 - Done
# Fetching links and content using selenium - Done
# Fetching links and page content from wordpress (page) apis - Done 


class ContentExtractor:
    
    def __init__(self) -> None:
        self.extracted_links:List[str] = []
          
    # Links extractor functions  
    def extract_links(self, url:str):
        if self.__is_sitemap(url):
            self.extracted_links = self._fetch_sitemap_links(url)
        else:
            url_type = self.__verify_url_type(url)
            if url_type == "Normal":
                self.extracted_links = self._extract_website_links(url)
            elif url_type == "Wordpress":
                self.extracted_links = self._extract_wordpress_links(url)
                
            self.extracted_links = [url] + self.extracted_links  
            
        print(f"TOTAL LINKS: {len(self.extracted_links) - 1}")
        return self.extracted_links

    def _fetch_sitemap_links(self, sitemap_url)-> List[str]:
        """Fetch all URLs from a sitemap."""
        try:
            print("Fetching links using sitemap........")
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
    
    def _extract_wordpress_links(self, url:str)-> List[str]:
        print(f"Extracting links of Wordpress website url: {url}")
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

    def _extract_website_links(self, url:str)-> List[str]:
        print(f"Trying to extract link from: {url}")

        # Try extracting with BeautifulSoup first
        if self.__is_needs_selenium(url):
            if links := self._extract_with_selenium(url, extract='links'):
                print("Extracted using Selenium")
                return links
        else:
            if links := self._extract_with_bs4(url, extract='links'):
                print("Extracted using BeautifulSoup")
                return links

        # If both methods fail, return an error
        print("Failed to extract content using both bs4 and Selenium.")
        return []


    # Content extractor functions
    def extract_content(self, urls:List[str])->dict[str, str]:
        print("Starting extracting content from urls...")
        results = {}
        
        if not urls:
            return results
        
        # limit the max limit of urls to extract
        MAX_URL_COUNT = 100
        if len(urls) >= MAX_URL_COUNT:
            urls = urls[:MAX_URL_COUNT]

        # Split the list into 5 equal parts
        num_threads = 5
        chunk_size = max(1, len(urls) // num_threads)  # Avoid division by zero
        url_chunks = [urls[i:i + chunk_size] for i in range(0, len(urls), chunk_size)]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(self._parallel_extract_content, chunk): chunk for chunk in url_chunks}

            for future in futures:
                results.update(future.result())  # Merge results from all threads
        return results        
        
    def _parallel_extract_content(self, urls:List[str]) -> dict[str, List[str]]:
        content = {}

        for url in urls:
            print(f"Trying to extract content from: {url}")
            
            # skip the login and sign in pages
            if ('login' in url) or ('sign' in url):
                continue
            
            url_type = self.__verify_url_type(url)
            
            if url_type == "Normal":
                content[url] = self._extract_website_content(url)
            elif url_type == 'Wordpress':
                content[url] = self._extract_wordpress_page_content(url)
            else:
                content[url] = None

        return content
        
        
    def _extract_wordpress_page_content(self, url:str):
        print(f"Extracting content of Wordpress website url: {url}")
        
        parsed_url = urlparse(url)
        
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}/"
        page_slug = url.strip('/').split('/')[-1]
        page_content = None
        
        # if the current url is the landing page of website
        # then we need to extract its content only
        if url == base_url:
            return self._extract_website_content(url)
        else:
            url_suffix = f"wp-json/wp/v2/pages?slug={page_slug}&_fields=slug,link,content"
            page_api_url = urljoin(base_url, url_suffix)
            response = requests.get(page_api_url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

            if response.status_code != 200:
                return None

            # Extract page URLs from JSON response
            content = response.json()[0]
            
            if 'content' in content:
                page_content = ''
                for field, value in content.items():
                    if field == 'content':
                        value = self.__clean_content(value.get('rendered', ''))
                    page_content += f"{field}:{value}\n"
            
            return page_content            
    
    def _extract_website_content(self, url:str):
        if self.__is_needs_selenium(url):
            if content := self._extract_with_selenium(url, extract='content'):
                print("Extracted using Selenium")
                return content
        else:
            if content := self._extract_with_bs4(url, extract='content'):
                print("Extracted using BeautifulSoup")
                return content

        # If both methods fail, return an error
        print("Failed to extract content using both BeautifulSoup and Selenium.")
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
                or (("X-Generator" in headers) and ("WordPress" in headers["X-Generator"]))
                or (("Link" in headers) and ('wp-json' in headers['Link']))):
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
            blocked_keywords = ["access denied", "forbidden", "blocked", "restricted", "enable javascript"]
            if any(keyword in soup.get_text().lower() for keyword in blocked_keywords):
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

        # if length of the visible content is lower than 100
        if len(self.__clean_content(response.text).split(" ")) < 100:
            return True
        
        return False
        
        
        
# start_time = time.time()
# extractor = ContentExtractor()
# links = extractor.extract_links("http://localhost:10003/")
# # # links = extractor.extract_links("https://www.bbc.com/sitemaps/https-sitemap-com-news-1.xml")
# content = extractor.extract_content(links)
# print("x....................", links)
# print("x....................", content)

# end_time = time.time()
# execution_time = end_time - start_time
# print(f"Execution Time: {execution_time:.4f} seconds")
