import re
import traceback
import asyncio
from typing import Literal, Optional
from langchain_chroma import Chroma
from openai import BaseModel
import uvicorn
import nest_asyncio
from fastapi import FastAPI
from rapidfuzz import fuzz, process

from chroma_handler import ChromaDBPopulator, DummyRetriever
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from langchain.chains import RetrievalQA
from langchain.schema import Document
import initial

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
    return tag

def entity_extractor(
        input:ChatInput, 
        division: Literal['database', 'document', 'website']
    ):
    doc = initial.NLP_PROCESSOR(input.message)

    # Extract named entities
    grammer_entities = ['NOUN', 'PROPN', 'ADJ']
    if division == "document":
        grammer_entities.append('VERB')
        
    entities = ""
    for ent in doc:
        if ent.pos_ in grammer_entities:
            entities += ent.text + " "
            print(f"ENTITIES: {ent.text} -> {ent.pos_}")
            
    entities = " ".join(list(set(ent.text for ent in doc if ent.pos_ in grammer_entities)))
    return entities or None
            
def filter_resultant_documents(
        input:ChatInput, 
        prompt:str, 
        retriever,
        division: Literal['database', 'document', 'website'],
        is_id_attached:bool, 
        user_id:Optional[int], 
    ):

    filtered_docs: list[Document] = []
    fallback_docs: list[Document] = []
    relevant_docs = retriever.get_relevant_documents(query=prompt)
    
    # extracting tags and entities for filtering
    entities = None
    if (division == "database" and (is_id_attached and user_id)) is False:
        entities = entity_extractor(input, division)
        
    for doc in relevant_docs:
        content = doc.page_content 
        
        if division == "database" and (is_id_attached and user_id):
            if not re.search(rf"\s{user_id}[,]", doc.page_content):
                continue

        if entities:
            best_match, score, _ = process.extractOne(
                entities, 
                [doc.page_content], 
                scorer=fuzz.WRatio
            )

            # Check if the best match is good enough
            threshold = initial.THRESHOLD[division]
            if not best_match or (best_match and score < threshold):
                content = None
            
        if content:
            filtered_docs.append(Document(
                page_content=content,
                metadata={}
            ))
        fallback_docs.append(Document(
            page_content=doc.page_content,
            metadata={}
        ))
    if division == 'database':
        print("VERIFIYING.,,...........", filtered_docs, fallback_docs)
    # send filtered docs
    if filtered_docs:
        return filtered_docs
    # send fallback docs removing metadata
    elif division != 'database':
        return fallback_docs
 
    # send blank docs
    blank_docs = [Document(page_content = "No data found",metadata = {})]
    return blank_docs

def retrieve_documents(        
        input:ChatInput, 
        prompt:str, 
        is_id_attached:bool, 
        user_id:int, 
        division: Literal['database', 'document', 'website']
    ):
    vector_store = initial.VECTOR_DB[division](initial.COLLECTION_NAME)
    
    search_kwargs= {
        'k':500 if division == 'database' else 350,
        # 'filter':  {'priority': division}
    }
    
    # filtering with tags
    filter_tag = None
    if division == 'database' and (filter_tag := filter_tags(input)): 
        print("FILTER TAG................", filter_tag)
        search_kwargs['filter'] = {'tags': filter_tag}
    
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
    
    filtered_documents = filter_resultant_documents(
        input, prompt, retriever, division, is_id_attached, user_id
    )
    return filtered_documents

async def retrieve_merged_documents(
        input:ChatInput, 
        prompt:str, 
        is_id_attached:bool, 
        user_id:int, 
    ):
    args = (input, prompt, is_id_attached, user_id)
    tasks = [
        asyncio.to_thread(retrieve_documents, *args, "database"),
        asyncio.to_thread(retrieve_documents, *args, "document"),
        asyncio.to_thread(retrieve_documents, *args, "website"),
    ]
    results = await asyncio.gather(*tasks)  # Run all fetches in parallel
    merged_documents = [doc for sublist in results for doc in sublist]  # Flatten the list
    print("MERGED RESULTS...................", merged_documents)
    return merged_documents

def get_re_search_retriever(query, merged_docs):
    temp_vector_store = Chroma.from_documents(merged_docs, initial.EMBEDDING_FUNCTION)  
    temp_retriever = temp_vector_store.as_retriever(search_type="mmr")
    refined_results = temp_retriever.get_relevant_documents(query)
    print("REFINED RESULTS ...................", refined_results)
    retriever = DummyRetriever(filtered_docs=refined_results)
    return retriever


from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
from langchain.schema import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer, util
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.preprocessing import MinMaxScaler
async def retrieve_response(
        input:ChatInput, 
        prompt:str, 
        is_id_attached:bool, 
        user_id:int, 
    ):
    try:     
        merged_documents = await retrieve_merged_documents(input, prompt, is_id_attached, user_id)
        # retriever = get_re_search_retriever(prompt, merged_documents)
        
        new_prompt, is_id_attached, user_id = attach_user_id(input)
        
        # Merge documents into a single string
        # context = "\n\n".join(f"- {doc.page_content}" for doc in merged_documents)
        context = [f"- {doc.page_content}" for doc in merged_documents]
        print("CONTEXT.......................", context)
        max_results=200
        # LLM prompt for filtering relevant documents
        filtering_prompt = f"""
        You are an AI assistant that extracts only the most relevant documents based on the user's query.
        You will be given multiple documents and a question. 
        Your task is to return only the most relevant documents with max words count {max_results}.
        If there are less than words count {max_results}, return only the available relevant ones.
        And don't explain about your answer, it will unnecessarily extend the response.

        --- DOCUMENTS START ---
        {context}
        --- DOCUMENTS END ---

        Question: {new_prompt}

        Now, return the most relevant documents with max words count {max_results}.
        """

        # Get the response from LLM
        # response = initial.FILTER_MODEL.invoke([
        #     SystemMessage(content="Extract and return the most relevant documents."), 
        #     HumanMessage(content=filtering_prompt)
        # ])
        # print("ANSSER................", response)
            
            
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Step 2: Tokenize documents for BM25
        tokenized_docs = [doc.lower().split() for doc in context]
        query_tokens = new_prompt.lower().split()

        # Step 3: Compute BM25 Scores
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = np.array(bm25.get_scores(query_tokens))

        # Step 4: Compute Embedding Similarity Scores
        doc_embeddings = model.encode(context, convert_to_tensor=True)
        query_embedding = model.encode(new_prompt, convert_to_tensor=True)
        embedding_scores = util.pytorch_cos_sim(query_embedding, doc_embeddings).cpu().numpy().flatten()

        # Step 5: Normalize Scores (Min-Max Scaling)
        scaler = MinMaxScaler()
        normalized_bm25 = scaler.fit_transform(bm25_scores.reshape(-1, 1)).flatten()
        normalized_embeddings = scaler.fit_transform(embedding_scores.reshape(-1, 1)).flatten()

        # Step 6: Combine Scores (Weighted Sum)
        bm25_weight = 0.5
        embedding_weight = 0.5
        final_scores = (bm25_weight * normalized_bm25) + (embedding_weight * normalized_embeddings)

        # Step 7: Retrieve Top N Documents
        top_n = 3  # Number of documents to return
        top_indices = np.argsort(final_scores)[::-1][:top_n]
        relevant_docs = [context[i] for i in top_indices]
        print("RELEVANT DOCS.................", relevant_docs)
        # Print Top Relevant Documents
        for idx, doc in enumerate(relevant_docs, 1):
            print(f"Rank {idx}: {doc}")
       
        # Create a RetrievalQA Chain
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm=initial.BASE_MODEL, 
        #     chain_type="stuff", 
        #     retriever=retriever,
        #     return_source_documents=True,
        # )
        
        # invoked_response = qa_chain.invoke({'query':prompt})
        # response = invoked_response.get('result', "No response available.")
        return "response"
    except Exception as e:
        print(f"Exception while retrieving response: {str(e)}")
        traceback.print_exc()
        return None

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

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
        
        response = await retrieve_response(input, final_prompt, is_id_attached, user_id)
        if response:
            # Summarizing new memory and saving it asynchronously
            asyncio.create_task(memory.add_memory(input.message, response))
        return {'response':response or initial.FALLBACK_MESSAGE}  
    except Exception:
        traceback.print_exc()
        return {"response": initial.FALLBACK_MESSAGE}




async def main():
    populator = ChromaDBPopulator()
    await populator.populate_chroma_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# Run the application
if __name__ == "__main__":
    # It allows to run nested event loops
    # first loop of asyncio and second one of uvicorn
    nest_asyncio.apply()
    asyncio.run(main())

