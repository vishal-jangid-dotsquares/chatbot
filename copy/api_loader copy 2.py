import asyncio
import os
from typing import Any, Dict, Literal, Optional, List
from bs4 import BeautifulSoup, Comment
from dotenv import load_dotenv
from woocommerce import API as WordpressApi
from langchain_core.documents import Document
import initial

load_dotenv()


PLATFORM_TYPES=Literal['wordpress', 'shopify']
WP_ENDPOINT_TYPES=Literal['posts', 'post_category', 'wp_users', 'wo_users', 'products', 'product_category', 'orders', 'cart']

class ApiLoader:
    
    _wp_params:Dict[
        WP_ENDPOINT_TYPES, 
        Dict[
            Literal['per_page', '_fields', 'status'],
            Optional[str|int]
        ]
    ] = {
        'wo_users':{
            'per_page': 1,
            '_fields': "id,name,link"
        },
        'wp_users':{
            'per_page': 1,
            '_fields': "id,name,link"
        },
        'posts':{
            'per_page': 100,
            '_fields': "title,content,status,author,date,link,categories"
        },
        'post_category':{
            'per_page': 100,
            '_fields': "name,link,description"
        },
        'products':{
            'per_page': 100,
            '_fields': "name,permalink,description,price,sale_price,on_sale,average_rating,categories,stock_status,images,attributes"
        },
        'cart':{
            "_fields":"total,line_items",
            "status":"checkout-draft"
        },
        'product_category':{
            'per_page':100,
            '_fields': 'name,description'
        },
        'orders':{
            'per_page': 100,
            "_fields":"status,date_created,discount_total,shipping_total,total,customer_id,billing,line_items",
            "status":"pending,processing,on-hold,completed,cancelled,refunded,failed"
        },
    }
    
    shopify_params = {}

    
    def __init__(self, platform:PLATFORM_TYPES, base_url:str, vectorstore = None) -> None:
        self.base_url = base_url.strip().rstrip('/')
        self.platform = platform
        self.vectorstore = vectorstore
            
    async def wp_data_loader(self, endpoints:List[WP_ENDPOINT_TYPES])->None:   
        tasks = [self._fetch_wp_data(endpoint) for endpoint in endpoints]
        await asyncio.gather(*tasks)
       
        print(f"Successfully fetched all the api endpoints: {endpoints}")
    
    async def shopify_loader(self):
        pass 


    async def _fetch_wp_data(self, endpoint:WP_ENDPOINT_TYPES):
        
        page_details = await self.__extract_total_wp_pages(endpoint)
        if not page_details:
            return None
        
        total_pages = page_details['total_pages']        
        total_items = page_details['total_items']
        
        # Fetch only parent products
        params = {}
        if endpoint == 'products':
            params["parent"] = 0
        
        response_data = []
        for page_count in range(1, int(total_pages)+1):
            response = await self._call_wp_api(
                endpoint, 
                params | {'page':page_count}
            )
            response_data.extend(response)
        return response_data

    async def _call_wp_api(self, endpoint:WP_ENDPOINT_TYPES, params=None, populate=True):
        
        response = await self.__execute_wp_api(endpoint, params)
        if not response:
            return []
        
        response = response.json()
        formatted_data = await self.__wp_data_formatter(endpoint, response)      
        if populate and self.vectorstore and formatted_data:
            documented_data = self.__wp_json_to_document(endpoint, formatted_data)
            await self.__populate_vector_db(documented_data)
        return formatted_data
            
    async def __execute_wp_api(self, endpoint:WP_ENDPOINT_TYPES, params=None):

        wordpress_endpoints:List[WP_ENDPOINT_TYPES] = ['posts', 'post_category', 'wp_users']
        woocommerce_endpoints:List[WP_ENDPOINT_TYPES] = ['products', 'product_category', 'orders', 'cart', 'wo_users']

        if endpoint in wordpress_endpoints:
            platform = 'woocommerce'
            wp_api_version = "wp/v2" 
            key=os.getenv('WOOCOMMERCE_CONSUMER_KEY')
            secret=os.getenv('WOOCOMMERCE_CONSUMER_SECRET')
        else:
            platform = 'wordpress'
            wp_api_version = "wc/v3"
            key=os.getenv('WORDPRESS_USERNAME')
            secret=os.getenv('WORDPRESS_PASSWORD')

            
        wcapi = WordpressApi(
            url=self.base_url,
            consumer_key=key,
            consumer_secret=secret,
            version=wp_api_version  
        )
        if platform == 'wordpress':
            wcapi.is_ssl = True
            wcapi.query_string_auth = False
            
        # params
        formatted_params = self._wp_params[endpoint] | (params or {})
        
        # endpoint
        updated_endpoint = endpoint
        if endpoint == 'product_category': 
            updated_endpoint = 'products/categories'
        elif endpoint == 'post_category': 
            updated_endpoint = 'categories'
        elif endpoint == 'cart':
            updated_endpoint = 'orders'
        elif endpoint in ['wp_users', 'wo_users']:
            updated_endpoint = 'users'
            
        res = wcapi.get(updated_endpoint, params=formatted_params)
        print("h......e...EE......", updated_endpoint, res.status_code, res.json())
        if res.status_code == 200:
            return res
        else:
            print(f"Unable to fetch data - endpoint:{endpoint} - {res.json()}")
            return None
          
          
    async def __wp_data_formatter(self, endpoint:WP_ENDPOINT_TYPES, data:List[Dict[str, Any]]):
        
        if endpoint == 'wo_users' or endpoint == 'wp_users':
            return await self.__wp_user_formatter(data)
        elif endpoint == 'cart':
            return await self.__wp_cart_formatter(data)
        elif endpoint == 'orders':
            return await self.__wp_order_formatter(data)
        elif endpoint == 'posts':
            return await self.__wp_post_formatter(data)
        elif endpoint == 'post_category':
            return await self.__wp_post_category_formatter(data)
        elif endpoint == 'products':
            return await self.__wp_product_formatter(data)
        elif endpoint == 'product_category':
            return await self.__wp_product_category_formatter(data)
        else:
            return data
        
    async def __wp_user_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return data
        
    async def __wp_cart_formatter(self, data: List[Dict[str, Any]]) -> Dict[str, Any]|List:
        if not data:
            return []
        
        if isinstance(data, list):
            copied_data = data[0]  # Extract first item if copied_data is a list

        line_items = copied_data.get('line_items', [])
        
        if not line_items:
            return copied_data  # Return early if no line items exist

        # Extract product IDs
        product_ids = ",".join(str(item.get("product_id")) for item in line_items if "product_id" in item)
        if not product_ids:
            return copied_data  # Return early if no valid product IDs found
        
        # Fetch product details asynchronously
        response = await self.__execute_wp_api('products', {
            '_fields': "name,permalink,price,images",
            "include": product_ids
        })
        if not response:
            return copied_data
        
        products = []
        response = response.json()
        for item in response:
            item['link'] = item.pop('permalink', None)
            if images := item.get("images"):
                item['image'] = images[0].get("src")
                item.pop('images', None)
            products.append(item)
        
        copied_data['products'] = products
        copied_data.pop('line_items', None) 
        return copied_data
            
    async def __wp_order_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Formats WooCommerce order data to a simplified structure.

        Args:
            data (List[Dict[str, Any]]): Raw WooCommerce order data.

        Returns:
            List[Dict[str, Any]]: Formatted order data.
        """
        formatted_orders = []
        
        for order in data:
            billing = order.get("billing", {})
            address_parts = [
                billing.get("address_1", ""), billing.get("city", ""), 
                billing.get("state", ""), billing.get("postcode", ""), billing.get("country", "")
            ]
            formatted_address = ", ".join(filter(None, address_parts))  # Remove empty values

            # Process products
            products = [
                {
                    "name": item.get("name", "").rstrip("."),  # Remove trailing period if present
                    "quantity": item.get("quantity", 1),
                    "price": str(item.get("subtotal", "0.00")),  # Ensure string type
                    "total": str(item.get("total", "0.00"))
                }
                for item in order.get("line_items", [])
            ]
            
            formatted_orders.append({
                "status": "success" if order.get("status") == "completed" else "pending",
                "currency": order.get("currency", "INR"),
                "date_created": order.get("date_created", ""),
                "discount_total": order.get("discount_total", "0.00"),
                "shipping_total": order.get("shipping_total", "0.00"),
                "total": order.get("total", "0.00"),
                "customer_id": order.get("customer_id", 0),
                "address": formatted_address,
                "products": products
            })

        return formatted_orders   
        
    async def __wp_post_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_posts = []
        
        for post in data:
            author_id = post.get("author") 
            response = await self.__execute_wp_api('wp_users', {'indclude':str(author_id)})

            formatted_posts.append({
                "date": post.get("date"),
                "link": post.get("link"),
                "title": post.get("title", {}).get("rendered", ""),
                "content": self.__clean_content(post.get("content", {}).get("rendered", "")),
                "author": response.json()[0] if response else author_id,
            })
        
        return formatted_posts
     
    async def __wp_post_category_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return data
    
    async def __wp_product_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted_products = []

        for product in data:
            formatted_product = {
                "name": product.get("name", ""),
                "link": product.pop("permalink", ""),
                "description": self.__clean_content(product.get("description", "")),
                "price": product.get("price", ""),
                "sale_price": product.get("sale_price", ""),
                "on_sale": product.get("on_sale", False),
                "stock_status": product.get("stock_status", ""),
                "average_rating": product.get("average_rating", "0.00"),
                "categories": ", ".join([category["name"] for category in product.get("categories", [])]),
                "image": product.get("images", [{}])[0].get("src", "") if product.get("images") else "",
                "attributes": [
                    {
                        "name": attr.get("name", ""),
                        "options": attr.get("options", [])
                    }
                    for attr in product.get("attributes", [])
                ]
            }
            
            formatted_products.append(formatted_product)

        return formatted_products

    async def __wp_product_category_formatter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return data
    
                
    async def __extract_total_wp_pages(self, endpoint:WP_ENDPOINT_TYPES):
        response = await self.__execute_wp_api(endpoint)
        if not response:
            return None 
        
        total_pages = response.headers.get("X-WP-TotalPages", '0')  # Extract total pages
        total_items = response.headers.get("X-WP-Total", '0')  # Extract total items
        
        data:Dict[Literal['total_pages', 'total_items'], str] = {
            'total_pages':total_pages, 
            'total_items':total_items
        }
        
        return data

    async def __populate_vector_db(self, documents:List[Document]):
        try:
            print("Starting adding documents parallely .........")
            
            num_threads = 5
            chunk_size = max(1, len(documents) // num_threads)  # Avoid division by zero
            document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

            async def populate_documents(vectorstore, document_chunks):
                """Populate vectorstore with documents asynchronously."""
                tasks = [vectorstore.aadd_documents(document) for document in document_chunks]
                await asyncio.gather(*tasks)  # Runs all coroutines concurrently

            # Calling all the 5 threads parallely
            # Check if an event loop is already running
            if asyncio.get_event_loop().is_running():
                await populate_documents(self.vectorstore, document_chunks)
            else:
                asyncio.run(populate_documents(self.vectorstore, document_chunks))
            return True
        except Exception as e:
            print(f"Error while adding documents - {str(e)}")
            return None
             
    def __wp_json_to_document(self, endpoint:WP_ENDPOINT_TYPES, data):
        if not isinstance(data, list):
            data = [data]
            
        documents = []
        for item in data:
            flatten_data = self.__flatten_dict(item, endpoint) 
            documents.append(
                Document(
                    page_content=", ".join(f"{key}:{value}" for key, value in flatten_data.items()),
                    metadata={
                        "devision": initial.DIVISIONS["db"], 
                        "source": self.platform, 
                        "tags": f"{endpoint.rstrip('s')}_tag"
                    }
                )
            )
        return documents
 
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
        
    def __flatten_dict(self, d, parent_key='', sep=' '):
        """Recursively flattens a nested dictionary into a single-level dict."""
        items = []
        for k, v in d.items():
            # remove trailing 's'
            parent_key = parent_key.title().rstrip('s')
            k = k.title().rstrip('s')
            
            new_key = f"{parent_key}{sep}{k}" if parent_key else k  # Create a unique key
            if isinstance(v, dict):
                items.extend(self.__flatten_dict(v, new_key, sep=sep).items())  # Recursively flatten dict
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    i += 1
                    if isinstance(item, dict):
                        items.extend(self.__flatten_dict(item, f"{new_key}_{i}", sep=sep).items())  # Flatten nested dicts inside lists
                    else:
                        items.append((f"{new_key}_{i}", item))  # Store list items as separate keys
            else:
                items.append((new_key, v))  # Store primitive values
        return dict(items)
    
               