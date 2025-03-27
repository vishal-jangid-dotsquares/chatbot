import os
import re
import traceback
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_core.messages import trim_messages, HumanMessage
import redis
from langchain.memory import  ConversationBufferWindowMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from chatbot.database import get_sql_agent
from chatbot.models import ChatInput
from chroma_handler import retrieve_response, ChromaDBPopulator
from chatbot.database import DatabaseConnector


# Initialize FastAPI
app = FastAPI()

# Load API Key from Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize the chat model
model = init_chat_model(
    "llama3-8b-8192", 
    model_provider="groq",
    max_tokens=100, 
    api_key=GROQ_API_KEY
)

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# fetch memory from redis memory
def get_memory(session_id: str):
    """Retrieve conversation memory from Redis"""
    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0"
    )
    return ConversationBufferWindowMemory(
        memory_key="chat_history",
        output_key="result",
        chat_memory=message_history, 
        return_messages=True,
        k=7 
    )

# Define the prompt template
PRE_PROMPTS = {
    'system': """
        - Always respond in **{language}**.  
        -{user_query} and give the answer in structured and readable format and in an intresting way but keep it short.
    """,
    "db_product": """
         **System Role:**  
            📌 **Database Rules for product table:**  
            The **products** table contains the following columns:  
            - **id (INTEGER, PRIMARY KEY)** – Unique product identifier.  
            - **name (TEXT)** – Product name.  
            - **category (TEXT)** – Product category (e.g., 'Clothing', 'Electronics').  
            - **price (REAL)** – Product price.  

            ✅ **Allowed Queries:**  
            - **Selection Queries (`SELECT`)** – Retrieve product details.  
            - **Filtering (`WHERE`)** – Search by category, name, or price range.  
            - **Sorting (`ORDER BY`)** – Order by price (highest/lowest) or name.  
            - **Pattern Matching (`LIKE`)** – Find products by partial name or category match.  

            ❌ **Restricted Queries:**  
            - **NO `UPDATE`, `DELETE`, `INSERT`, or `DROP` operations.**  
            - **DO NOT return raw SQL queries in responses.**  

            ⚡ **Best Practices:**  
            - **Limit results to 5** unless the user specifies a different limit.  
            - **Optimize queries** using indexed columns (`id`, `category`).  
            - **Provide direct and relevant answers** without unnecessary possibilities.  

            ---

            📌 **Example Queries & Responses:**  

            ✅ **User Input:** _List all products in the 'Electronics' category._  
            ➡ **AI Response:** _"Here are 5 products in Electronics: Laptop ($1200), Smart TV ($899), ..."_

            ✅ **User Input:** _Show the cheapest products._  
            ➡ **AI Response:** _"Here are 5 cheapest products: T-Shirt ($9.99), Mouse ($15), ..."_

            ✅ **User Input:** _How many products are in the 'Clothing' category?_  
            ➡ **AI Response:** _"There are 42 products in the Clothing category."_

            ✅ **User Input:** _What are the most expensive laptops?_  
            ➡ **AI Response:** _"Here are the top 5 expensive laptops: MacBook Pro ($2499), ..."_

            ---

            🚀 **Now, provide a user query, and I will fetch the most relevant products!**  

        User Query: {user_query}
        """
}

def get_products_agent(user_query):
    """LangChain agent generates SQL queries for SQLite automatically."""
    
    sql_agent = get_sql_agent(model)
    stream = sql_agent.invoke({"action": "sql_db_query", "input": user_query})
    return stream

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""
    try:
        # Retrieve memory for the session
        memory = get_memory(input.session_id)
        pattern = r'\b(?:show|list|find|get|search|fetch|retrieve|display)\b'
        
        if re.search(pattern, input.message,re.IGNORECASE):
            prompt = PRE_PROMPTS["db_product"].format(user_query = input.message)
            response = get_products_agent(prompt)
            return {'response': response.get('output', 'No matching products available!')}
        else:
            qa_chain = retrieve_response(input.message, model, memory)
            prompt = PRE_PROMPTS['system'].format(user_query = input.message, language=input.language)
            retrieved_answer = qa_chain.invoke({'query':input.message})
            memory.chat_memory.add_user_message(input.message)
            memory.chat_memory.add_ai_message(retrieved_answer.get("result", "No response available."))
            return {"response": retrieved_answer.get('result') or "No response available."}

    except Exception:
        traceback.print_exc()
        return {"response": 'Sorry, i am unable to find any valid results'}


# Run the application
if __name__ == "__main__":
    import uvicorn
    # populator = ChromaDBPopulator()    
    # populator.populate_chroma_db()

    uvicorn.run(app, host="0.0.0.0", port=8000)
