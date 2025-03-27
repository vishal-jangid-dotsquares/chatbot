import json
import os
from typing import Callable, Dict, Literal
from dotenv import load_dotenv
import redis
import spacy
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load API Key from Environment Variable
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY_9413")

# Available models
MODELS : Dict[
    Literal['vision','specdec',"versatile", "guard"], str
    ] = {
    "vision" : "llama-3.2-90b-vision-preview",
    "specdec" : "llama-3.3-70b-specdec",
    "versatile" : "llama-3.3-70b-versatile",
    "guard": "llama3-8b-8192"
}

# Initialize the chat model
FILTER_MODEL = init_chat_model(
    MODELS['guard'],
    model_provider="groq",
    api_key=GROQ_API_KEY,
)
BASE_MODEL = init_chat_model(
    MODELS['vision'],
    model_provider="groq",
    api_key=GROQ_API_KEY,
)

# Initialising embedding function
EMBEDDING_FUNCTION = HuggingFaceEmbeddings()
# EMBEDDING_FUNCTION.model_name = "BAAI/bge-large-en-v1.5"
EMBEDDING_FUNCTION.show_progress =True


# Load the existing config file and vector database
def GET_CONFIGS(key:str):
    try:
        with open("config.json", "r") as file:
            config = json.load(file)
            return config.get(key)
    except FileNotFoundError:
        raise ValueError(" Config file not found!")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file!")

COLLECTION_NAME = 'vishal'

DIVISIONS: Dict[
        Literal['db', 'doc', 'web'],
        Literal['database', 'document', 'website']
    ] = {
    'db':'database',
    'doc':'document',
    'web':'website'
}


def VECTOR_STORE(directory_name:str) -> Callable[[str], Chroma]:
    vectorstore:Chroma = Chroma(
        persist_directory=directory_name, 
        embedding_function=EMBEDDING_FUNCTION
    )
    
    def set_collection(collection_name:str)->Chroma:
        vectorstore._chroma_collection = vectorstore._client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata=None,
        )
        return vectorstore
    
    return set_collection

VECTOR_DB: Dict[
    Literal['document', 'website', 'database'], 
    Callable[[str], Chroma]
    ] = {
    'document': VECTOR_STORE('chroma_db_directory/document_vector_db'),
    'website': VECTOR_STORE('chroma_db_directory/website_vector_db'),
    'database': VECTOR_STORE('chroma_db_directory/database_vector_db'),
}

# Redis client setup
REDIS_CLIENT = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Intializin Natural language processor
NLP_PROCESSOR = spacy.load("en_core_web_sm")

# Rapid Fizz Entity Matching Score
THRESHOLD : dict[
    Literal[
        "document", "database", "website"
    ], int] = {
    "document" : 50,
    "database" : 50,
    "website" : 50
}

# Define the prompt template
PRE_PROMPTS:Dict[Literal['system', 'division'], str] = {
    'system': """
        Always respond in english. And give the answer in structured, human readable format, 
        in an intresting, add few icons to make it more understandable, keep it short, 
        to the point and length should not exceed 100 words. 
        Remember, whenver you can't find any valid answer, always reply with a 'didn't find any valid results', 
        respond very politely and request to try another question. 
        ---------------------------------------------------
        ### New User query: 
        {user_query}

        ### Conversation History: 
        {history}
    """,
    'division': """
        You are an AI assistant that classifies user queries into one of three categories based on intent, context, and emotion.  
        Additionally, you should understand the user’s emotions—whether they are **curious, frustrated, confused, or making a general inquiry**—to improve classification accuracy.  

        ---

        ### **1. 'document'** → Select this category if the user is looking for **guidance, policies, FAQs, or official instructions**.  
        **Common scenarios:**  
        a. Policies, guidelines, instructions, rules, limitations, or regulations.  
        b. Complaints, **customer support**, service-related doubts, or **feature clarifications**.  
        c. **FAQs** or commonly asked questions about how something works.  
        d. If a user provides an **incomplete phrase** and asks for an explanation.  
        e. **Ordering process:** How to place an order, available **payment methods, refund policies, return/exchange processes**.  
        f. **Legal or security-related queries** (e.g., "How is my data protected?" or "What are the terms of use?").  

        **Examples:**  
        - *"Can I order by phone?"* →
        - *"Can I order by phone?"* → **document** ✅ (This is about the ordering process, not a specific order.)  
        - *"How do I return a product?"* → **document** ✅ (Policy-related question.)  
        - *"What is your refund policy?"* → **document** ✅ (Asking for rules and policies.)  
        - *"What does 'out of stock' mean?"* → **document** ✅ (User is confused about a phrase.)  

        ---

        ### **2. 'database'** → Select this category if the user is asking for **data related to orders, products, carts, categories, or user-specific information**.  
        **Common scenarios:**  
        a. Information about **their personal orders, carts, products, or past purchases**.  
        b. **Details about items**: lists, prices, varieties, availability, or specifications of orders, carts, products, categories, or services.  
        c. **Counting requests**: total number of orders, items in cart, product categories, available discounts, or stock.  
        d. **Sales and promotions**: user wants to know about discounts or ongoing offers.  
        e. **Product/service tracking**: checking an order status, shipment tracking, or estimated delivery time.  

        **Examples:**  
        - *"Where is my order?"* → **database** ✅ (Asking for personal order tracking.)  
        - *"How many items are in my cart?"* → **database** ✅ (Asking for specific user data.)  
        - *"Show me all available laptops under $1000."* → **database** ✅ (Requesting product details.)  
        - *"What is the price of iPhone 15 Pro?"* → **database** ✅ (Asking for product pricing.)  
        - *"Is there any sale on shoes?"* → **database** ✅ (User wants to know about discounts.)  

        ---

        ### **3. 'website'** → Select this category if the user is asking for **general website-related or publicly available information**.  
        **Common scenarios:**  
        a. **Company details**: services, about us, mission statement, or contact information.  
        b. **Website policies & guidelines** (specific to the website, not general policies).  
        c. **Blog-related queries**: asking for articles, blog posts, publishing date, author names, or links.  
        d. **Customer reviews, testimonials, or external feedback.**  
        e. **General knowledge, factual inquiries, or people-related questions.**  
        f. **Incomplete or unclear phrases** where the user is asking for clarification.  

        **Examples:**  
        - *"Who is the author of the blog 'The Importance of Time Management'?"* → **website** ✅ (Asking about blog details.)  
        - *"What services does your company provide?"* → **website** ✅ (Company-related question.)  
        - *"Can you share customer reviews?"* → **website** ✅ (User wants to see feedback/testimonials.)  
        - *"Tell me about Nikola Tesla."* → **website** ✅ (General knowledge question.)  
        - *"Explain the phrase 'time is money'."* → **website** ✅ (User wants phrase explanation.)  

        ---

        ### **Emotion & Context Awareness**  
        If the user query contains frustration (e.g., *"Why is my order late?"*), confusion (e.g., *"I don't understand the return policy."*), or urgency (e.g., *"Where is my order? It's delayed!"*), prioritize a more precise classification:  
        - **Frustration + Order Tracking → 'database'
        - **Frustration + Order Tracking → 'database'** (Example: *"Why is my order late?"*)  
        - **Confusion + Policies → 'document'** (Example: *"I don't understand the return policy."*)  
        - **Urgency + Order Issue → 'database'** (Example: *"Where is my order? It's delayed!"*)  
        - **Curiosity + Blog Details → 'website'** (Example: *"Who wrote the article on time management?"*)  

        ---

        ### **📌 Final Classification Prompt**
        **Use the following classification system to categorize user queries:**

        #### **User Query:**  
        ```{user_query}```  

        **Respond with only one category name:**  
        - `'database'` → If the query is about personal orders, product details, stock, pricing, carts, totals, or discounts.  
        - `'document'` → If the query is about policies, FAQs, order/refund/return process, or general guidance.  
        - `'website'` → If the query is about the company, services, blogs, author names, links, customer feedback, or general knowledge.  

    """
}

# Falback message
FALLBACK_MESSAGE = 'Sorry, i am unable to find any valid results. Please, try with another question 😊'


# Keyword extractor patterns
USER_PATTERN: Dict[
    Literal[
        'digit_pattern', 
        'entity_pattern', 
        'exclued_pattern'
        ], str
    ] = {
    "digit_pattern" : r'(\d|my)',
    "entity_pattern" : r'(orde?(?:rs?|re?d)|p(?:e|u)rcha?ses?d?|buy|bought|carts?|products?|cancelled)',
    "exclued_pattern" : r'(cancel)'
}

GREETING_PATTERNS: Dict[
    Literal['greeting_pattern', 'strict_gretting_pattern'], str
    ] = {
    "greeting_pattern" : r'\b(greetings?|good\s?(?:morning|evening|afternoon)|gm|i?\s?need help)\b',
    "strict_gretting_pattern" : r'^(hello|hi|hey)$'
}  
    
FILTER_TAG_PATTERNS: Dict[
    Literal[
        'order_pattern', 
        'product_pattern',
        'product_category_pattern', 
        'category_pattern', 
        'continue_pattern'
        ], str
    ] = {
    "order_pattern" : r'\b(orde?(?:rs?|re?d)|p(?:e|u)rcha?ses?d?|buy|bought)\b',
    "product_pattern" : r'\b(products?|items?)\b',
    "product_category_pattern" : r'\b(t(?:y|i)pes?|v(?:e|a)r(?:i|ei|ie)t(?:y|i|is|ies?|eis?)|c(?:e|a)t(?:e|i|ie|ei)g(?:a|o)r(?:y|i|e|ee|eis?|ies?)|kinds?)\b',
    "category_pattern" : r'\b(c(?:e|a)t(?:e|i|ie|ei)g(?:a|o)r(?:y|i|e|ee|eis?|ies?))\b',
    "continue_pattern" : r'\b(continues?|more|last|previous|next|go|go ahe?a?d)\b'
}

CHROMA_FILTER_PATTERNS: Dict[
    Literal[
        'order_pattern', 
        'product_pattern', 
        'category_pattern'
        ], str
    ] = {
    "order_pattern" : r'\b(orders?|purchase(?:s|d)?|orders?details?|sales?|sales?detail?)\b',
    "product_pattern" : r'\b(products?|items?|products?details?|categoryproducts?)\b',
    "category_pattern" : r'\b(categor(?:y|ies)|products?(?:types?|categor(?:y|ies))|categor(?:y|ies)details?)\b'
}

