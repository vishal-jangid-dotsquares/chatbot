import asyncio
import json
from typing import Dict, Literal, Optional, List
from bs4 import BeautifulSoup, Comment
import requests

from chatbot.database import DatabaseConnector


class ApiLoader:
    
    _wordpress_params:Dict[
            Literal['posts', 'users', 'products', 'orders', 'cart'], 
            Dict[
                Literal['per_page', '_fields', 'status'],
                Optional[str|int]
            ]
        ] = {
        'users':{
            'per_page': 1,
            '_fields': "id,name,description"
        },
        'posts':{
            'per_page': 100,
            '_fields': "slug,title,content,status,author,date,link,categories"
        },
        'products':{
            'per_page': 100,
            '_fields': "slug,permalink,description,price,sale_price,on_sale,average_rating,categories,tags,stock_status"
        },
        'orders':{
            'per_page': 100,
            "_fields":"status,currency,date_created,discount_total,shipping_total,total,customer_id,billing,line_items",
            "status":"pending,processing,on-hold,completed,cancelled,refunded,failed"
        },
        'cart':{
            'per_page':1,
            "_fields":"status,currency,date_created,discount_total,shipping_total,total,customer_id,billing,line_items",
            "status":"checkout-draft"
        }
    }
    
    shopify_params = {}

    
    def __init__(self, platform:Literal['wordpress', 'shopify'], base_url:str, userId, vectorstore) -> None:
        self.platform:Literal['wordpress', 'shopify'] = platform
        self.base_url = base_url.strip().rstrip('/')
        self.vectorstore = vectorstore
        self.userId = userId
        
        # Initialise database cursor
        self.connector = DatabaseConnector('mysql')
        self.conn = self.connector.connect()
        self.cursor = self.conn.cursor()
        
    async def load(self, endpoints:List[Literal['posts', 'users']]):
        if self.platform == 'shopify':
            await self.shopify_loader()
        else:
            return await self.wordpress_loader(endpoints)
            
    async def wordpress_loader(self, endpoints:List[Literal['posts', 'users']]):      
        return await self._fetch_parallel(endpoints)
    
    async def _fetch_parallel(self, endpoints:List[Literal['posts', 'users']]):
        tasks = [self._fetch_data(endpoint) for endpoint in endpoints]
        results = await asyncio.gather(*tasks)
        result_data = {endpoints[i]:results[i] for i in range(len(endpoints))}
        print("resutls................", result_data)
        print(f"Successfully fetched all the api endpoints: {endpoints}")
        return result_data
    
    async def _fetch_data(self, endpoint:Literal['posts', 'users']):
        formatted_url = self.__format_wordpress_url(endpoint)
        
        page_details = await self.__extract_total_wordpress_pages(formatted_url)
        if not page_details:
            return None
        
        total_pages = page_details['total_pages']        
        total_items = page_details['total_items']
        
        response_data = []
        for page_count in range(1, int(total_pages)+1):
            formatted_url += f"&page={page_count}" 
            response = await self.__execute_api(formatted_url)
            if not response:
                continue
            
            await self.__populate_database(endpoint, response)
            response_data.extend(response)
        return response_data
            
    async def __execute_api(self, url:str):
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code != 200:
            return None 
        json_response = response.json()
        
        # Cleaning the posts content
        if 'post' in url:
            for item in json_response:
                if "content" in item and "rendered" in item["content"]:
                    item["content"] = self.__clean_content(item["content"]["rendered"])
                
        return json_response
            
    def __format_wordpress_url(self, endpoint:Literal['posts', 'users']):
        url = f'{self.base_url}/{endpoint}'
        params = '&'.join([f'{key}={value}' for key, value in self._wordpress_params[endpoint].items()])
        
        url_with_params = f'{url}?{params}'
        return url_with_params
    
    async def __extract_total_wordpress_pages(self, url) -> Dict[Literal['total_pages', 'total_items'], str]|None:
        response = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})

        if response.status_code != 200:
            return None 
        
        total_pages = response.headers.get("X-WP-TotalPages", '0')  # Extract total pages
        total_items = response.headers.get("X-WP-Total", '0')  # Extract total items
        
        data:Dict[Literal['total_pages', 'total_items'], str] = {
            'total_pages':total_pages, 
            'total_items':total_items
        }
        return data
    
    async def __populate_database(self, endpoint:Literal['posts', 'users'], response:List[Dict]):
        # Insert data into MySQL
        try:
            
            data = [(endpoint, self.userId, json.dumps(json_data)) for json_data in response]
            
            insert_query = """
            INSERT INTO wordpress_data (endpoint, user_id, data) 
            VALUES (%s, %s, %s)
            """
            
            self.cursor.executemany(insert_query, data)
            self.conn.commit()
        except Exception as e:
            print(f"Exception while populating mysql database - endpoint: {endpoint} \n : {str(e)}")
            self.conn.rollback()
            self.connector.close()

          
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
        
    
    
    async def shopify_loader(self):
        pass            
            