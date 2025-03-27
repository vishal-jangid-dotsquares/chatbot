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
        You are an AI assistant that classifies user queries into one of three categories:
        1. 'document' → If the user has queries related to:
            a. Policies, guidelines, instructions, rules, or limitations.
            b. Complaints, customer support, service-related doubts, or feature clarifications.
            c. FAQs or commonly asked questions about how something works.
            d. Incomplete phrases where the user seeks an explanation.
            e. Questions related to **how to place an order**, available payment options, refund policies, or return processes.

        2. 'database' → If the user has queries related to:
            a. Their personal orders, carts, products, or posts.
            b. Lists, prices, varieties, or details of orders, carts, products, categories, or posts.
            c. The total count of orders, carts, products, categories, or posts.
            d. Discounts, promotions, or sales on products.

        3. 'website' → If the user has queries related to:
            a. Company information, services, or general "About Us" details.
            b. Policies or guidelines specific to the company's website or platform.
            c. Blogs, page links, customer feedback, or testimonials.
            d. People, facts, or general knowledge questions.
            e. Incomplete phrases where the user seeks clarification.
            f. Explain about some phrases

        ### **User Query:**
        {user_query}

        Respond with only one category name: 'database', 'document', or 'website'.
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

