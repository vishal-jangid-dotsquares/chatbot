import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from chroma_handler import retrieve_response, ChromaDBPopulator
from chatbot.database import DatabaseConnector


# Initialize FastAPI
app = FastAPI()

# Load API Key from Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize the chat model
model = init_chat_model(
    # "llama-3.2-90b-vision-preview", 
    # "llama-3.3-70b-specdec", 
    # "llama-3.3-70b-versatile", 
    "qwen-2.5-coder-32b",
    model_provider="groq",
    api_key=GROQ_API_KEY
)

# Initializing memory
memory = CustomChatMemory()


# Define the prompt template
PRE_PROMPTS = {
    'system': """Always respond in englist. And give the answer in structured, human readable format, in an intresting, add some emojis to make it intresting, keep it short, to the point and length should not exceed 50 words. Remember, whenver you can't find any valid answer, always reply with a Sorry, responsd very politely and request to try another question. User query: {user_query}""",
}

PATTERNS = {
    "digit_pattern" : r'(\d|my)',
    "entity_pattern" : r'(order|orders|ordered|purchase|purchases|purchased|buy|bought|cart|product|products|cancelled)',
    "exclued_pattern" : r'(cancel)',
    "greeting_pattern" : r'\b(greetings|good morning|good evening|good afternoon|i need help|need help)\b',
    "strict_gretting_pattern" : r'^(hello|hi|hey)$'
}

async def handle_greetings(input:ChatInput, memory):
    message = input.message
    response = None
    if (re.search(PATTERNS['greeting_pattern'], message)
        or re.search(PATTERNS['strict_gretting_pattern'], message)
    ):
        response = "👋 Hello! 😊 How can I help you today? 🤔"
    elif 'nice to meet you' in message:
        response = "🙂 Nice to meet you too! 👋 How can I assist you today? 🤔"
    elif 'how are you' in message:
        response = "😊 I'm good, thanks! 👍 Ready to help 🤔"

    if response:
        asyncio.create_task(memory.add_memory(message, response))
    return response
    
async def attach_necessary_information(input:ChatInput, memory):
    user_id = 14 #2
    digit_pattern = r'[\d|my]'
    entity_pattern = r'[order|orders|ordered|purchase|purchases|purchased|buy|bought|cart|product|products|cancelled]'
    exclued_pattern = r'[cancel]'
    is_id_attached = False
    if (user_id 
        and re.search(digit_pattern, input.message) 
        and re.search(entity_pattern, input.message)
        and not re.search(exclued_pattern, input.message)
        ):
        input.message += f' and where customer id is {user_id} or user id is {user_id}'
        is_id_attached = True
    
    # Attaching memory and preprompt to the input
    new_prompt = await memory.attach_memory_to_prompt(input.message)
    print("NEW PROMPT...........", new_prompt)
    prompt = PRE_PROMPTS['system'].format(user_query = new_prompt)
    print("Final PROMPT...........", prompt)
    return prompt, is_id_attached

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""
    try:
        memory.add_user(input.session_id)
        input.message = input.message.lower()
        
        # Handle greeting prompts
        if response := (await handle_greetings(input, memory)):
            return {'response':response}
        
        # Attaching required informations
        prompt, is_id_attached = await attach_necessary_information(input, memory)

        with ThreadPoolExecutor() as executor:
            if is_id_attached:
                future_high = executor.submit(retrieve_response, prompt, model, "high")
                response = future_high.result()
            else:
                # Run both requests in parallel
                response = ""
                future_low = executor.submit(retrieve_response, prompt, model, "low")
                future_high = executor.submit(retrieve_response, prompt, model, "high")

                futures = {future_low: "low", future_high: "high"}

                for future in as_completed(futures):
                    response = future.result()
                    
                    if "sorry" not in response.lower():
                        break  # Return the first successful response

                # If both contain "sorry", return the "high" response
                if "sorry" in response.lower():
                    response = future_high.result()
            
            # Summarizing new memory and saving it asynchronously
            asyncio.create_task(memory.add_memory(input.message, response))
            return {'response':response}  
    except Exception:
        traceback.print_exc()
        return {"response": 'Sorry, i am unable to find any valid results. Please, try with another question 😊'}


# Run the application
if __name__ == "__main__":
    import uvicorn
    populator = ChromaDBPopulator()    
    populator.populate_chroma_db()

    uvicorn.run(app, host="0.0.0.0", port=8000)
