import os
import re
import traceback
import asyncio
from typing import Literal
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from fastapi import FastAPI
import redis
import spacy
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from chroma_handler import ChromaDBPopulator, vector_store
from langchain.chat_models import init_chat_model
from langchain.chains import RetrievalQA
from langchain.schema import Document, BaseRetriever


load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Load API Key from Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY_9413")
print("API KEY.............", GROQ_API_KEY)

# Initialize the chat model
model = init_chat_model(
    "llama-3.2-90b-vision-preview", 
    # "llama-3.3-70b-specdec", 
    # "llama-3.3-70b-versatile", 
    # "qwen-2.5-coder-32b",
    model_provider="groq",
    api_key=GROQ_API_KEY
)

# Initializing memory
memory = CustomChatMemory()

# Load the existing vector database
vectorstore = vector_store()

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# Intializin Natural language processor
nlp = spacy.load("en_core_web_sm")

# Define the prompt template
PRE_PROMPTS = {
    'system': """Always respond in english. And give the answer in structured, human readable format, in an intresting, add 2-3 emojis to make it intresting, keep it short, to the point and length should not exceed 50 words. Remember, whenver you can't find any valid answer, always reply with a Sorry, respond very politely and request to try another question. User query: {user_query}""",
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
    
def attach_user_id(input:ChatInput):
    user_id = 14 #2
    digit_pattern = r'(\d|my)'
    entity_pattern = r'(order|orders|ordered|purchase|purchases|purchased|buy|bought|cart|product|products|cancelled)'
    exclued_pattern = r'(cancel)'
    is_id_attached = False
    if (user_id 
        and re.search(digit_pattern, input.message) 
        and re.search(entity_pattern, input.message)
        and not re.search(exclued_pattern, input.message)
    ):
        input.message += f' and where customer id is {user_id} or user id is {user_id}'
        is_id_attached = True
    
    
    return input.message, is_id_attached, user_id

def filter_tags(input:ChatInput):
    order_pattern = r'\b(orders?|ordered|purchases?|purchased|buy|bought)\b'
    product_pattern = r'\b(products?|items?)\b'
    category_pattern = r'\b(types?|variety|vareity|varieties|vareities|vareiti|variti|kinds?)\b'
    continue_pattern = r'\b(continues?|more|last|previous|next|go|go ahead)\b'
    
    redis_tag_key = f"{input.session_id}_tag_key"
    tag = ''
    if re.search(order_pattern, input.message, re.IGNORECASE):
        tag = 'order_tag'
    elif (re.search(category_pattern, input.message, re.IGNORECASE) 
        and re.search(product_pattern, input.message, re.IGNORECASE)):
        tag = 'category_tag'
    elif re.search(product_pattern, input.message, re.IGNORECASE):
        tag = 'product_tag'
    elif re.search(continue_pattern, input.message, re.IGNORECASE):
        if extracted_tag := redis_client.get(redis_tag_key):
            tag = extracted_tag
    
    if tag:
        redis_client.set(redis_tag_key, tag)
    print("TAG.......................", input.message, tag)
    return tag

def entity_extractor(input:ChatInput):
    doc = nlp(input.message)

    # Extract named entities
    entities = []
    for ent in doc:
        if ent.pos_ in ['NOUN', 'PROPN', 'ADJ']:
            entities.append(ent.text)
            print(f"{ent.text} -> {ent.pos_}")
    # entities = list(set(ent.text for ent in doc if ent.pos_ in ['NOUN', 'PROPN', 'ADJ']))
    entity_pattern = r'\b(?:' + '|'.join(map(re.escape, entities)) + r')\b'
    return entities, entity_pattern
            
def enhanced_retriever(input, prompt, retriever, priority, user_id, is_id_attached):
    filtered_docs: list[Document] = []
    relevant_docs = retriever.get_relevant_documents(query=prompt)
    
    # extracting tags and entities for filtering
    tag = ''
    if priority == 'high':
        tag = filter_tags(input)

    entities = None
    entity_pattern = ''
    if priority == 'high' and not tag:
        entities = entity_extractor(input)
            

    for doc in relevant_docs:
        content = None 
        
        if priority == "high":
            if (is_id_attached 
                and user_id
                and str(user_id) in doc.page_content
                and tag in doc.metadata.get('tags', 'None')
            ):
                content = doc.page_content
            elif tag in doc.metadata.get('tags', 'None'):
                content = doc.page_content
            elif entities:
                if re.search(entity_pattern, doc.page_content, re.IGNORECASE):
                    content = doc.page_content
            # else:
            #     content = doc.page_content
        else:
            content = doc.page_content
        
        print('---------------------------------------------------------------------------------------------------------------')
        if content:
            print(f"**********",doc.metadata)
            filtered_docs.append(Document(
                page_content=content,
                metadata={}
            ))
        print(doc.metadata)
        print('---------------------------------------------------------------------------------------------------------------')
    
    class DummyRetriever(BaseRetriever):
        filtered_docs: list[Document] #add type hinting.

        def _invoke(self, query: str) -> list[Document]:
            return self.filtered_docs

        def _get_relevant_documents(self, query: str) -> list[Document]:
            return self.filtered_docs

    retriever = DummyRetriever(filtered_docs=filtered_docs)
    return retriever

def retrieve_response(input, prompt, is_id_attached, user_id, llm, priority:Literal['high', 'low']):
    search_kwargs= {
        'k':100 if priority == 'high' else 30,
        'filter':  {'priority': priority}
    }

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    retriever = enhanced_retriever(input, prompt, retriever, priority, user_id, is_id_attached)
    
    # Create a RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    invoked_response = qa_chain.invoke({'query':prompt})
    if priority == 'high':
        print("INVOKED RESPONSE...............", invoked_response)
    response = invoked_response.get('result', "No response available.")
    return response



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
        prompt_with_id, is_id_attached, user_id = attach_user_id(input)
        prompt_with_memory = await memory.attach_memory_to_prompt(prompt_with_id)
        prompt = PRE_PROMPTS['system'].format(user_query = prompt_with_memory)

        with ThreadPoolExecutor() as executor:
            if is_id_attached:
                future_high = executor.submit(retrieve_response, input, prompt, is_id_attached, user_id, model, "high")
                response = future_high.result()
            else:
                # Run both requests in parallel
                response = ""
                future_low = executor.submit(retrieve_response, input, prompt, is_id_attached, user_id, model, "low")
                future_high = executor.submit(retrieve_response, input, prompt, is_id_attached, user_id, model, "high")

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
