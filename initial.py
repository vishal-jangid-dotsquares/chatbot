import json
import os
import re
from typing import Callable, Dict, List, Literal
from dotenv import load_dotenv
import redis.asyncio as redis
import spacy
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.language_models import BaseChatModel


# ** THIRD PARTY RELATED INITIALS **
# Load API Key from Environment Variable
load_dotenv()

# Redis client setup
REDIS_CLIENT = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Intializin Natural language processor
NLP_PROCESSOR = spacy.load("en_core_web_sm")



# ** MODEL RELATED INITIALS **
GROQ_API_KEY = os.getenv("GROQ_API_KEY_9413")

# Available models
MODEL_NAMES : Dict[
    Literal['vision','specdec',"versatile", "small"], str
] = {
    "vision" : "llama-3.2-90b-vision-preview",
    "specdec" : "llama-3.3-70b-specdec",
    "versatile" : "llama-3.3-70b-versatile",
    "small": "llama3-8b-8192"
}

# Initialize the chat model
def initialise_model(model:Literal[
    'vision','specdec',"versatile", "small"
]) ->BaseChatModel:
    return init_chat_model(
        MODEL_NAMES[model],
        model_provider="groq",
        api_key=GROQ_API_KEY,
    )

MODELS : Dict[
    Literal['vision','specdec',"versatile", "small"], 
    BaseChatModel
] = {
    "vision" : initialise_model('vision'),
    # "specdec" : initialise_model('specdec'),
    # "versatile" : initialise_model('versatile'),
    "small": initialise_model('small')
}



# ** EMBEDDING FUNCTION RELATED INITIALS **
# Initialising embedding function
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {
    "device": "cpu",  # Force CPU usage
    "trust_remote_code": True  # If using custom transformers from Hugging Face
}
encode_kwargs = {
    "batch_size": 8,  # Reduce batch size to avoid high RAM usage
    "normalize_embeddings": True,  # Normalize for better cosine similarity search
}

EMBEDDING_FUNCTION = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
EMBEDDING_FUNCTION.show_progress = True



# ** VECTOR DB RELATED INITIALS **
def VECTOR_STORE(directory_name:str) -> Callable[[str], Chroma]:
    vectorstore:Chroma = Chroma(
        persist_directory=directory_name, 
        embedding_function=EMBEDDING_FUNCTION,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    def set_collection(collection_name:str)->Chroma:
        vectorstore._chroma_collection = vectorstore._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
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




# ** IMPORTANT STATIC INITIALS ** 
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

PLATFORM_NAME: Literal['wordpress', 'shopify', 'base'] = 'wordpress'

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
PRE_PROMPTS:Dict[
    Literal['followUp', 'memory', 'system', 'division'], str
] = {
    'followUp':"""
        You are an AI system designed to determine whether a new user query is a **strict follow-up** to a previous query.

        ### **Strict Definition of a Follow-up Query**
        A query is a follow-up **ONLY IF** it meets **ALL** of these conditions:
        1. **It explicitly references something mentioned in the previous query.**
        2. **It cannot be answered meaningfully without first knowing the previous query.**
        3. **It directly depends on details or context provided in the previous query.**
        4. **If the new query is asked alone, it would be unclear or incomplete.**

        ### **Strict Rejection Criteria**
        A query is **NOT** a follow-up if:
        - It introduces a **new topic or request** that was not explicitly stated in the previous query.
        - It **can be understood and answered independently** without needing the previous query.
        - It refers to **a specific entity (e.g., "VITA") that is not explicitly mentioned** in the previous query.
        - It is **only loosely related but does not depend on** the previous query.

        ---
        **Previous Query:** "{prev_query}"  
        **New Query:** "{current_query}"  
        ---

        ### **Instructions**
        - **ONLY respond with "yes" or "no".**  
        - **Do NOT assume** a follow-up if there is any ambiguity. If unsure, default to **"no"**.  
        - **If the new query can be fully understood on its own, respond with "no"** even if the topics are similar.  

    """,
    'memory': """
        Summarize the following conversation in **no more than 300 words** while keeping key details and maintaining clarity. 

        Your summary should:
        - Retain **60% of the previous summary** to ensure continuity.
        - Incorporate **40% of the current conversation** to capture recent updates.
        - Be clear, concise, and not exceed the word limit.

        Previous Summary:
        {old_summary}

        New Conversation:
        {input_text}

        Generate an updated summary while ensuring coherence and readability.
        Only give the summarise conversation don't explain anything related to the process.
    """,
    'system': """
        ### AI Chatbot Assistant for {company}  
        *Managed by Vishal*  

        - **Language:** Always respond in English.  
        - **Format:** Keep answers structured, engaging, and under 100 words. Use icons for clarity if needed.  
        - **Product & Content Queries:** Suggest relevant options from provided data or given past conversation. If a link exists, include it.  
        - For personal orders, use only **documented details with user/customer ID** (⚠ Never reveal IDs).  

        ### **Follow-Up Handling**  
        - if last question exists, check if the new question follows the last question.  
        - Example:  
        - **User:** "Show me the latest laptops."  
        - **User (Follow-Up):** "Which has the best battery?"  
        - **Bot:** "The [Model] has the best battery. [Link]"  

        ### **Understanding User Emotions**  
        Respond accordingly:  
        - 🎉 Excited: "Great choice! Here’s the best option..."  
        - 🤔 Confused: "No worries! Let me simplify..."  
        - 🙏 Frustrated: "I’m here to help! Let’s find a solution..."  

        ### **No Valid Answer?**  
        - Reply: "I couldn’t find a match. Try a different query!"  
        - Never generate responses beyond the given context or previous conversations (except for greetings or introductions).  

        ### **Guidelines**  
        ✔ Search & prioritize results based on user intent and emotions.
        ✔ Answer only what is asked, avoiding unnecessary details.
        ✔ Do not disclose extra information unless the user explicitly asks.

        **New Query:** {current_question}  
        **Previous (if any):** {last_question}  
        **History:** {history}  
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
        f. **Blog/Post-related queries**: asking for articles, blog posts, publishing date, author names, or links. 
        g. **Category-based recommendations** (e.g., suggesting products, categories, or blog posts). 

        **Examples:**  
        - *"Where is my order?"* → **database** ✅ (Asking for personal order tracking.)  
        - *"How many items are in my cart?"* → **database** ✅ (Asking for specific user data.)  
        - *"Show me all available laptops under $1000."* → **database** ✅ (Requesting product details.)  
        - *"What is the price of iPhone 15 Pro?"* → **database** ✅ (Asking for product pricing.)  
        - *"Is there any sale on shoes?"* → **database** ✅ (User wants to know about discounts.)  
        - *"I love watches, do you have watches?"* → **database** ✅ (Checking product availability.)  
        - *"Who is the author of the blog/post 'The Importance of Time Management'?"* → **website** ✅ (Asking about blog details.)  
        - *"My wife’s anniversary is coming, suggest me a ring for a gift."* → **database** ✅ (Seeking product recommendation.)  

        ---

        ### **3. 'website'** → Select this category if the user is asking for **general website-related or publicly available information**.  
        **Common scenarios:**  
        a. **Company details**: services, about us, links, mission statement, or contact information.  
        b. **Website policies & guidelines** (specific to the website, not general policies).  
        c. **Customer reviews, testimonials, or external feedback.**  
        d. **General knowledge, factual inquiries, or people-related questions.**  
        e. **Incomplete or unclear phrases** where the user is asking for clarification.  

        **Examples:**  
        - *"What services does your company provide?"* → **website** ✅ (Company-related question.)  
        - *"Can you share customer reviews?"* → **website** ✅ (User wants to see feedback/testimonials.)  
        - *"Tell me about Nikola Tesla."* → **website** ✅ (General knowledge question.)  
        - *"Explain the phrase 'time is money'."* → **website** ✅ (User wants phrase explanation.)  

        ---

        ### **Emotion & Context Awareness**  
        If the user query contains frustration (e.g., *"Why is my order late?"*), confusion (e.g., *"I don't understand the return policy."*), or urgency (e.g., *"Where is my order? It's delayed!"*), prioritize a more precise classification:  
        - **Frustration + Order Tracking → 'database'
        - **Frustration + Order Tracking → 'database'** (Example: *"Why is my order late?"*)  
        - **Confusion/Curiosity + Policies → 'document'** (Example: *"I don't understand the return policy."*)  
        - **Urgency + Order Issue → 'database'** (Example: *"Where is my order? It's delayed!"*)  
        - **Curiosity + Blog/Posts Details → 'database'** (Example: *"Who wrote the article on time management?"*)  

        ---

        ### **📌 Final Classification Prompt**
        **Use the following classification system to categorize user queries:**

        #### **User Query:**  
        ```{user_query}```  

        **Respond with only one category name:**  
        - `'database'` → If the query is about personal orders, blogs, posts, product details, stock, pricing, carts, suggestions, totals, or discounts.  
        - `'document'` → If the query is about policies, FAQs, order/refund/return/exchange process, or general guidance.  
        - `'website'` → If the query is about the company, services, links, customer feedback, general knowledge, or website-related topics.  .  

    """
}

# Falback message
FALLBACK_MESSAGE = 'Sorry, i am unable to find any valid results. Please, try with another question 😊'



# ** PATTERN RELATED INITIALS **

# Keyword extractor patterns
USER_PATTERN: Dict[
    Literal[
        'self_reffering_pattern', 
        'entity_pattern', 
        'exclued_pattern'
        ], str
    ] = {
    "self_reffering_pattern" : r'\b(my|my last|give me my)',
    "entity_pattern" : r'(orde?(?:rs?|re?d)|p(?:e|u)rcha?ses?d?|placed?|buy|bought|carts?|cancelled)',
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
        'helper_category_pattern'
        ], str
    ] = {
    'post_pattern': r'\b(posts?|(?:b|v)log(?:s?|ings?)|articles?|authors)\b',
    'cart_pattern': r'\b(carts?|buckets?|saved? (?:items?|product?))\b',
    "order_pattern" : r'\b(orde?(?:rs?|re?d)|p(?:e|u)rcha?ses?d?|buy|bought)\b',
    "product_pattern" : r'\b((?:products?|items?)\s*(?:under|c(?:e|a)t(?:e|i|ie|ei)g(?:a|o)r(?:y|i|e|ee|eis?|ies?)))\b',
    "helper_category_pattern" : r'\b(t(?:y|i)pes?|v(?:e|a)r(?:i|ei|ie)t(?:y|i|is|ies?|eis?)|c(?:e|a)t(?:e|i|ie|ei)g(?:a|o)r(?:y|i|e|ee|eis?|ies?)|kinds?|catalogs?|catalouges?)\b'
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

# Pre compiled follow pattern
FOLLOW_UP_PATTERN:List = list(re.compile(pattern, re.IGNORECASE) for pattern in  [
    r"\b(next|anymore|continues?|another|else|extend|next|go (?:aheads?|on)?)\b",  # Single-word implicit follow-ups
    r"\b((?:what|how) about|(?:any|any others?|anything|something)\s*(?:alternatives?|else)\s*(than (?:this|these|that|those|them))?)\b",  # Comparative follow-ups
    r"\b((?:tell|shows?|gives?|lists?|explains?)\s*(:?me|me in|it|in|(?:this|these|that|those|them|it) in)?\s*(?:deep|more|short|shorter))\b",  # Requesting more details
    r"\b((?:ok|yes)?\s*(?:extend|shows?|gives?|lists?|explains?) (?:this|these|that|those|them|its?)|expand on that|continue with more)\b",  # Requesting more details
    r"\b(and\?|what else|any others?|(?:lasts?|recents?|previous)\s*(?:question|responses?|results?|answers?))\b",  # Implicit contextual follow-ups
    r"\b(do you have other|show me different|what else do you offer|similar products?|alternative brands?|explain more about this)\b",
    r"\b((?:go aheads?|continues?|keep going|can you continue|any suggestions)\s*(on (?:this|these|that|those|them|its?)?))\b"
])


