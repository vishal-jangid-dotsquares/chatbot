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
FILTERING_MINIMUM_SCORE : dict[
    Literal[
        "document", "database", "website"
    ], int] = {
    "document" : 30,
    "database" : 30,
    "website" : 30
}

# Define the prompt template
PRE_PROMPTS:Dict[Literal['system', 'division'], str] = {
    'system': """
        You are an AI chatbot assistant working for {company}, Vishal is your owner and creator.
        Always respond in english. And give the answer in structured, human readable format, 
        in an intresting, add few icons to make it more understandable, keep it short, 
        to the point and length should not exceed 100 words. 
        If the context contain any valid link related to the question, always add it in the answer. 
        Remember, whenver you can't find any valid answer, always reply with a 'didn't find any valid results', 
        respond very politely and request to try another question. 
        Always give response from the given context or previous conversation, if you don't find any valid result never give answers from your own (except greeting or intro).
        Always give only specific answer related to the user question, don't reveal any extra information till the user ask about it.
        ---------------------------------------------------
        ### New User query: 
        {user_query}

        ### Conversation History: 
        {history}
    """,
    'division': """
        You are an AI assistant that classifies user queries into one of three categories based on intent, context, and emotion.  
        Additionally, you should recognize when users are asking for **suggestions** (products, categories, or posts) and classify accordingly.  

        ---

        ### **1. 'document'** → Select this category if the user is looking for **guidance, policies, FAQs, or official instructions**.  
        **Common scenarios:**  
        - **Policies, guidelines, or rules** (e.g., refund policies, return/exchange processes, security policies).  
        - **Customer support inquiries** (e.g., complaints, service-related doubts, feature clarifications).  
        - **FAQs** (e.g., “How does this work?”, “What is your refund policy?”).  
        - **Ordering process details** (e.g., “How do I place an order?”, “What payment methods do you accept?”).  
        - **Legal or security-related queries** (e.g., “How is my data protected?”).  

        **Examples:**  
        - *"How do I return a product?"* → **document** ✅ (Policy-related question.)  
        - *"What are your refund rules?"* → **document** ✅ (Policy clarification.)  
        - *"What does 'out of stock' mean?"* → **document** ✅ (User is confused about a phrase.)  

        ---

        ### **2. 'database'** → Select this category if the user is asking for **data related to orders, products, categories, blogs, or user-specific information**.  
        **Common scenarios:**  
        - **Personal data requests** (e.g., orders, carts, purchases, account details).  
        - **Product/service details** (e.g., availability, stock, prices, specifications).  
        - **Counting requests** (e.g., “How many items are in my cart?”, “How many laptops do you have?”).  
        - **Sales and promotions** (e.g., “Are there any discounts on shoes?”).  
        - **Order tracking and shipment details** (e.g., “Where is my order?”, “When will my package arrive?”).  
        - **Blog/Post-related queries** (e.g., retrieving stored blog posts, asking for authors, publishing dates).  
        - **Category-based recommendations** (e.g., suggesting products, categories, or blog posts).  

        **Examples:**  
        - *"Where is my order?"* → **database** ✅ (Tracking personal order.)  
        - *"How many items are in my cart?"* → **database** ✅ (Requesting cart details.)  
        - *"Show me all available laptops under $1000."* → **database** ✅ (Requesting product details.)  
        - *"Do you have blog posts on AI?"* → **database** ✅ (Asking for stored posts.)  
        - *"Suggest a good smartwatch."* → **database** ✅ (Product recommendation request.)  
        - *"I love watches, do you have watches?"* → **database** ✅ (Checking product availability.)  
        - *"My wife’s anniversary is coming, suggest me a ring for a gift."* → **database** ✅ (Seeking product recommendation.)  

        ---

        ### **3. 'website'** → Select this category if the user is asking for **general website-related or publicly available information**.  
        **Common scenarios:**  
        - **Company-related details** (e.g., “What services does your company provide?”, “Tell me about your company.”).  
        - **Website-specific policies & guidelines** (e.g., “What are the terms of service?”).  
        - **Customer reviews, testimonials, or external feedback.**  
        - **General knowledge, factual inquiries, or people-related questions.**  

        **Examples:**  
        - *"What services does your company provide?"* → **website** ✅ (Company-related question.)  
        - *"Can you share customer reviews?"* → **website** ✅ (User wants testimonials.)  
        - *"Tell me about Nikola Tesla."* → **website** ✅ (General knowledge question.)  
        - *"Explain the phrase 'time is money'."* → **website** ✅ (Phrase explanation.)  

        ---

        ### **🧠 Emotion & Context Awareness**  
        If the query contains **frustration**, **confusion**, or **urgency**, adjust classification accordingly:  
        - **Frustration + Order Tracking → 'database'** (e.g., *"Why is my order late?"*).  
        - **Confusion + Policies → 'document'** (e.g., *"I don't understand the return policy."*).  
        - **Urgency + Order Issue → 'database'** (e.g., *"Where is my order? It's delayed!"*).  
        - **Curiosity + Blog/Post Details → 'database'** (e.g., *"Who wrote the article on time management?"*).  

        ---

        ### **📌 Final Classification Prompt**  
        **Use the following classification system to categorize user queries:**

        #### **User Query:**  
        ```{user_query}```  

        **Respond with only one category name:**  
        - `'database'` → If the query is about orders, products, categories, stock, pricing, carts, totals, discounts, blogs, or suggestions.  
        - `'document'` → If the query is about policies, FAQs, return/refund/exchange process, or general guidance.  
        - `'website'` → If the query is about company info, customer reviews, general knowledge, or website-related topics.  

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
        'post_pattern', 
        'cart_pattern', 
        'order_pattern', 
        'product_pattern',
        'helper_category_pattern', 
        'continue_pattern'
        ], str
    ] = {
    'post_pattern': r'\b(posts?|(?:b|v)log(?:s?|ings?)|articles?|authors)\b',
    'cart_pattern': r'\b(carts?|buckets?|saved? (?:items?|product?))\b',
    "order_pattern" : r'\b(orde?(?:rs?|re?d)|p(?:e|u)rcha?ses?d?|buy|bought)\b',
    "product_pattern" : r'\b(products?|items?)\b',
    "helper_category_pattern" : r'\b(t(?:y|i)pes?|v(?:e|a)r(?:i|ei|ie)t(?:y|i|is|ies?|eis?)|c(?:e|a)t(?:e|i|ie|ei)g(?:a|o)r(?:y|i|e|ee|eis?|ies?)|kinds?)\b',
    "continue_pattern" : r'\b(continues?|more|last|previous|next|go|go ahe?a?d)\b'
}

CHROMA_FILTER_PATTERNS: Dict[
    Literal[
        'order_pattern', 
        'product_pattern', 
        'product_category_pattern',
        'post_category_pattern'
        ], str
    ] = {
    "order_pattern" : r'\b(orders?|purchase(?:s|d)?|orders?details?|sales?|sales?detail?)\b',
    "product_pattern" : r'\b(products?|items?|products?details?|categoryproducts?)\b',
    "product_category_pattern" : r'\b(categor(?:y|ies)|products?(?:types?|categor(?:y|ies))|categor(?:y|ies)details?)\b'
}

