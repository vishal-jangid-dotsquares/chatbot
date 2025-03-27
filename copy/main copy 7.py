import re
import traceback
import asyncio
import uvicorn
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI

from chroma_handler import ChromaDBPopulator, DummyRetriever
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from langchain.chains import RetrievalQA
from langchain.schema import Document
import initial

from rapidfuzz import fuzz, process

# Initialize FastAPI
app = FastAPI()

async def handle_greetings(input:ChatInput, memory):
    message = input.message

    greeting_pattern = initial.GREETING_PATTERNS['greeting_pattern']
    strict_gretting_pattern = initial.GREETING_PATTERNS['strict_gretting_pattern']
    
    response = None
    if (re.search(greeting_pattern, message)
        or re.search(strict_gretting_pattern, message)
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
    digit_pattern = initial.USER_PATTERN['digit_pattern']
    entity_pattern = initial.USER_PATTERN['entity_pattern']
    exclued_pattern = initial.USER_PATTERN['exclued_pattern']
    
    is_id_attached = False
    if (user_id 
        and re.search(digit_pattern, input.message) 
        and re.search(entity_pattern, input.message)
        and not re.search(exclued_pattern, input.message)
    ):
        input.message += f' and where customer/user id is {user_id}'
        is_id_attached = True
    
    return input.message, is_id_attached, user_id

def filter_tags(input:ChatInput):
    order_pattern = initial.FILTER_TAG_PATTERNS['order_pattern']
    product_pattern = initial.FILTER_TAG_PATTERNS['product_pattern']
    product_category_pattern = initial.FILTER_TAG_PATTERNS['product_category_pattern']
    category_pattern = initial.FILTER_TAG_PATTERNS['category_pattern']
    continue_pattern = initial.FILTER_TAG_PATTERNS['continue_pattern']
    
    redis_tag_key = f"{input.session_id}_tag_key"
    tag = None
    if (
        (re.search(product_category_pattern, input.message, re.IGNORECASE) 
        and re.search(product_pattern, input.message, re.IGNORECASE))
        or re.search(category_pattern, input.message, re.IGNORECASE)
    ):
        tag = 'category_tag'
    elif re.search(product_pattern, input.message, re.IGNORECASE):
        tag = 'product_tag'
    elif re.search(order_pattern, input.message, re.IGNORECASE):
        tag = 'order_tag'
    elif re.search(continue_pattern, input.message, re.IGNORECASE):
        if extracted_tag := initial.REDIS_CLIENT.get(redis_tag_key):
            tag = extracted_tag
    
    if tag:
        initial.REDIS_CLIENT.set(redis_tag_key, tag)
    print("FILTER TAG AND MESSAGE.......................", input.message, tag)
    return tag

def entity_extractor(input:ChatInput, priority):
    doc = initial.NLP_PROCESSOR(input.message)

    # Extract named entities
    grammer_entities = ['NOUN', 'PROPN', 'ADJ']
    if priority == "low":
        grammer_entities.append('VERB')
        
    entities = ""
    for ent in doc:
        if ent.pos_ in grammer_entities:
            entities += ent.text + " "
            print(f"ENTITIES: {ent.text} -> {ent.pos_}")
            
    entities = " ".join(list(set(ent.text for ent in doc if ent.pos_ in grammer_entities)))
    print(entities)
    return entities or None
            
def enhanced_retriever(*args):
    input, prompt, retriever, priority, tag, user_id, is_id_attached = args

    filtered_docs: list[Document] = []
    relevant_docs = retriever.get_relevant_documents(query=prompt)
    
    # extracting tags and entities for filtering
    entities = None
    if priority == 'low' or priority == 'high' and tag is None:
        entities = entity_extractor(input, priority)
    
    print("x.....................", relevant_docs)
    for doc in relevant_docs:
        content = doc.page_content 
        
        if priority == "high":
            if is_id_attached and user_id:
                if not re.search(rf"\s{user_id}[,]", doc.page_content):
                    content = None
                    print("NOT ATTACHED.....................................")

        if entities:
            searchable_content = doc.page_content
            if priority == "low":
                searchable_content = doc.page_content.split("Answer")[0]
                
            best_match, score, _ = process.extractOne(entities, [searchable_content], scorer=fuzz.WRatio)

            if priority == "high":
                print("BEST MATCH..........", priority, score, best_match)
            # Check if the best match is good enough
            threshold = initial.THRESHOLD[priority]
            if not best_match or (best_match and score < threshold):
                content = None
                print("NO ENTITIY.....................")
            
        print('---------------------------------------------------------------------------------------------------------------')
        if content:
            print("**********",content)
            filtered_docs.append(Document(
                page_content=content,
                metadata={}
            ))
        print(doc.page_content)
        print(doc.metadata)
        print('---------------------------------------------------------------------------------------------------------------')

    if not filtered_docs:
        filtered_docs.append(Document(
            page_content = "No data found",
            metadata = {}
        ))
    
    retriever = DummyRetriever(filtered_docs=filtered_docs)
    return retriever

def retrieve_response(*args):
    try:
        input, prompt, is_id_attached, user_id, priority = args
        
        # Fetching required vector instance
        if priority == "high":
            vector_directory = 'database'
        else:
            vector_directory = 'document'
        vector_store = initial.VECTOR_DB[vector_directory](initial.COLLECTION_NAME)
        
        
        search_kwargs= {
            'k':500 if priority == 'high' else 100,
            'filter':  {'priority': priority}
        }
        
        # filtering with tags
        tag = None
        if priority == 'high': 
            if tag := filter_tags(input):
                search_kwargs['filter'] = {'tags': tag}

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        retriever = enhanced_retriever(input, prompt, retriever, priority, tag, user_id, is_id_attached)
        
        # Create a RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=initial.BASE_MODEL, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
        )
        invoked_response = qa_chain.invoke({'query':prompt})
        response = invoked_response.get('result', "No response available.")
        return response
    except Exception as e:
        print(f"Exception while retrieving response: {str(e)}")
        traceback.print_exc()
        return None

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

    fallback_message = 'Sorry, i am unable to find any valid results. Please, try with another question 😊'
    try:
        # Initializing memory
        memory = CustomChatMemory()
        memory.add_user(input.session_id)
        
        input.message = input.message.lower()
        
        # Handle greeting prompts
        if response := (await handle_greetings(input, memory)):
            return {'response':response}
        
        # Attaching required informations - user_id, memory, pre_prompt
        new_prompt, is_id_attached, user_id = attach_user_id(input)
        
        conversation_history = await memory.get_conversation()
        final_prompt = initial.PRE_PROMPTS['system'].format(
            user_query = new_prompt,
            history = conversation_history
        )

        params = (input, final_prompt, is_id_attached, user_id)
        if is_id_attached:
            response = retrieve_response(*params, "high")
        else:
            response = retrieve_response(*params, "low")
            if response and ("valid" in response):
                response = retrieve_response(*params, "high")
                
        if response:
            # Summarizing new memory and saving it asynchronously
            asyncio.create_task(memory.add_memory(input.message, response))
        return {'response':response or fallback_message}  
    except Exception:
        traceback.print_exc()
        return {"response": fallback_message}



nest_asyncio.apply()

async def main():
    populator = ChromaDBPopulator()
    await populator.populate_chroma_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    

# Run the application
if __name__ == "__main__":
    asyncio.run(main())

