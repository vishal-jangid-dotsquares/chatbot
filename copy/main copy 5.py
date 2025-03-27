import re
import traceback
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

from fastapi import FastAPI
from chroma_handler import ChromaDBPopulator, DummyRetriever
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from langchain.chains import RetrievalQA
from langchain.schema import Document
import initial


# Initialize FastAPI
app = FastAPI()
vector_store = initial.VECTOR_STORE()

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
    category_pattern = initial.FILTER_TAG_PATTERNS['category_pattern']
    continue_pattern = initial.FILTER_TAG_PATTERNS['continue_pattern']
    
    redis_tag_key = f"{input.session_id}_tag_key"
    tag = None
    if re.search(order_pattern, input.message, re.IGNORECASE):
        tag = 'order_tag'
    elif (re.search(category_pattern, input.message, re.IGNORECASE) 
        and re.search(product_pattern, input.message, re.IGNORECASE)):
        tag = 'category_tag'
    elif re.search(product_pattern, input.message, re.IGNORECASE):
        tag = 'product_tag'
    elif re.search(continue_pattern, input.message, re.IGNORECASE):
        if extracted_tag := initial.REDIS_CLIENT.get(redis_tag_key):
            tag = extracted_tag
    
    if tag:
        initial.REDIS_CLIENT.set(redis_tag_key, tag)
    print("TAG.......................", input.message, tag)
    return tag

def entity_extractor(input:ChatInput):
    doc = initial.NLP_PROCESSOR(input.message)

    # Extract named entities
    entities = []
    for ent in doc:
        if ent.pos_ in ['NOUN', 'PROPN', 'ADJ']:
            entities.append(ent.text)
            print(f"{ent.text} -> {ent.pos_}")
    # entities = list(set(ent.text for ent in doc if ent.pos_ in ['NOUN', 'PROPN', 'ADJ']))
    entity_pattern = r'\b(?:' + '|'.join(map(re.escape, entities)) + r')\b'
    return entities, entity_pattern
            
def enhanced_retriever(*args):
    input, prompt, retriever, priority, tag, user_id, is_id_attached = args

    filtered_docs: list[Document] = []
    relevant_docs = retriever.get_relevant_documents(query=prompt)
    
    # extracting tags and entities for filtering
    entities = None
    entity_pattern = ''
    if priority == 'high' and tag is None:
        entities, entity_pattern = entity_extractor(input)
            
    for doc in relevant_docs:
        content = None 
        
        if priority == "high":
            if (is_id_attached and user_id
                and re.search(rf'\b\'{str(user_id)}\'\b', doc.page_content)
                and tag and tag in doc.metadata.get('tags', 'None')
            ):
                content = doc.page_content
            elif tag and tag in doc.metadata.get('tags', 'None'):
                content = doc.page_content
            elif entities:
                if re.search(entity_pattern, doc.page_content, re.IGNORECASE):
                    content = doc.page_content
                    print("IN ENTITIY...........", entity_pattern, entities)
            # else:
            #     content = doc.page_content
        else:
            content = doc.page_content
        
        print('---------------------------------------------------------------------------------------------------------------')
        if content:
            if priority == "high":
                print(f"**********",content)
                print(f"**********",doc.metadata)
            filtered_docs.append(Document(
                page_content=content,
                metadata={}
            ))
        if priority == "high":
            print(doc.metadata)
        print('---------------------------------------------------------------------------------------------------------------')

    retriever = DummyRetriever(filtered_docs=filtered_docs)
    return retriever

def retrieve_response(*args):
    try:
        input, prompt, is_id_attached, user_id, priority = args
        
        search_kwargs= {
            'k':500 if priority == 'high' else 30,
            'filter':  {'priority': priority}
        }
        
        # filtering with tags
        tag = None
        if priority == 'high' and (tag := filter_tags(input)):
            if tag:
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
        if priority == 'high':
            print("INVOKED RESPONSE...............", invoked_response)
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
        prompt_with_id, is_id_attached, user_id = attach_user_id(input)
        prompt_with_memory = await memory.attach_memory_to_prompt(prompt_with_id)
        prompt = initial.PRE_PROMPTS['system'].format(user_query = prompt_with_memory)

        with ThreadPoolExecutor() as executor:
            params = (input, prompt, is_id_attached, user_id)
            if is_id_attached:
                future_high = executor.submit(retrieve_response, *params, "high")
                response = future_high.result()
            else:
                # Run both requests in parallel
                response = ""
                future_low = executor.submit(retrieve_response, *params, "low")
                future_high = executor.submit(retrieve_response, *params, "high")

                futures = {future_low: "low", future_high: "high"}
                for future in as_completed(futures):
                    response = future.result()
                    
                    if response and "valid" not in response.lower():
                        break  # Return the first successful response

                # If both contain "valid", return the "high" response
                if response and "valid" in response.lower():
                    response = future_high.result()
            
            if response:
                # Summarizing new memory and saving it asynchronously
                asyncio.create_task(memory.add_memory(input.message, response))
                
            return {'response':response or fallback_message}  
    except Exception:
        traceback.print_exc()
        return {"response": fallback_message}


# Run the application
if __name__ == "__main__":
    import uvicorn
    populator = ChromaDBPopulator()
    populator.populate_chroma_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
