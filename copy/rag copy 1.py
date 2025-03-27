import re
import traceback
import asyncio
from typing import List, Literal, Optional
from langchain_chroma import Chroma
from rapidfuzz import fuzz, process

from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from langchain.chains import RetrievalQA
from langchain.schema import Document, BaseRetriever
from langchain.schema import SystemMessage, HumanMessage
import initial


    
class DummyRetriever(BaseRetriever):
    filtered_docs: list[Document] #add type hinting.

    def _invoke(self, query: str) -> list[Document]:
        return self.filtered_docs

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.filtered_docs
    

class Rag:
    userId:Optional[int] = None
    is_userId_attached:Optional[bool] = False
    pre_prompt_message:str = ""
    
    def __init__(self, input:ChatInput) -> None:
        self.input:ChatInput = input
        self.message = input.message.lower()
        
        # Initialise memory
        self.memory:CustomChatMemory = CustomChatMemory()
        self.memory.add_user(self.input.session_id)
     
    async def invoke(self):
        try:           
            # Handle greeting prompts
            greeting_response = self.__handle_greetings()
            if greeting_response:
                return greeting_response
            
            response = await self._retrieve_response()
            if response:
                # Summarizing new memory and saving it asynchronously
                asyncio.create_task(
                    self.memory.add_memory(
                        self.message, 
                        response
                    )
                )
            return response or initial.FALLBACK_MESSAGE
        except Exception:
            traceback.print_exc()
            return initial.FALLBACK_MESSAGE   
        
    async def _retrieve_response(self):
        # Attaching user id and preprompt
        self.__attach_userId()
        self.pre_prompt_message = await self.__attach_pre_prompt()
        
        retriever = await self._smart_retriever()
     
        # Create a RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=initial.BASE_MODEL, 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
        )
        
        invoked_response = qa_chain.invoke({'query':self.pre_prompt_message})
        response = invoked_response.get('result', "No response available.")
        return response
        # return 'response'            

    async def _smart_retriever(self):
        # it retriever parallely
        # first it retireve devision type
        # and then it retrive all the documents parallely
        if self.is_userId_attached:
            retriever = await self._get_retriever("database")
            division = "database"
        else:
            results = await asyncio.gather(
                self.__find_division_type(),
                self._retrieve_documents_parallely()
            )
            division, merged_retrievers = results
            print("retrieved DIVIOSN...............", division)
            
            if 'database' in division:
                division = 'database'
                retriever = merged_retrievers[0]
            elif 'document' in division:
                division = 'document'
                retriever = merged_retrievers[1]
            else:
                division = 'website'
                retriever = merged_retrievers[2]
            
             
        relevant_docs = retriever.invoke(input=self.message)   
        filtered_documents = self._filter_documents(division, relevant_docs)
        print("RELEVANT DOCS....................",division, relevant_docs)
        print("SMART RETRIEVER RESULTES.................", filtered_documents)       
        
        # Re-filtering the documents
        retriever = await self._re_filter_retriever(filtered_documents)
        return retriever

    async def _retrieve_documents_parallely(self):
        results = await asyncio.gather(
            self._get_retriever("database"),
            self._get_retriever("document"),
            self._get_retriever("website")
        )
        return results

    async def _get_retriever(self, division: Literal['database', 'document', 'website']):
        vector_store = initial.VECTOR_DB[division](initial.COLLECTION_NAME)
        
        search_kwargs= {
            'k':500 if division == 'database' else 350,
        }
        
        # filtering with tags
        filter_tag = None
        if division == 'database':
            if filter_tag := self.__filter_tags(): 
                print("FILTER TAG................", filter_tag)
                search_kwargs['filter'] = {'tags': filter_tag}
            
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        return retriever
            
    def _filter_documents(
            self, 
            division: Literal['database', 'document', 'website'],
            documents: List[Document]
        ):
        
        filtered_docs: list[Document] = []
        fallback_docs: list[Document] = []
                
        # extracting tags and entities for filtering
        entities = None
        if (division == "database" and (self.is_userId_attached and self.userId)) is False:
            entities = self.__entity_extractor(division)
        
        relevant_docs = documents
        for doc in relevant_docs:
            content = doc.page_content 
            
            if division == "database"and (self.is_userId_attached and self.userId):
                if not re.search(rf"\s{self.userId}[,]", doc.page_content):
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

        # send filtered docs
        if filtered_docs:
            return filtered_docs
        # send fallback docs removing metadata
        elif division != 'database':
            return fallback_docs
    
        # send blank docs
        blank_docs = [Document(page_content = "No data found",metadata = {})]
        return blank_docs
    
    async def _re_filter_retriever(self, merged_docs):
        temp_vector_store = Chroma.from_documents(merged_docs, initial.EMBEDDING_FUNCTION)  
        temp_retriever = temp_vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs= {'k':15}
        )
        refined_results = temp_retriever.invoke(self.message)
        
        retriever = DummyRetriever(filtered_docs=refined_results)
        return retriever
    
        
    def __attach_userId(self):
        self.userId = 14 #2
        digit_pattern = initial.USER_PATTERN['digit_pattern']
        entity_pattern = initial.USER_PATTERN['entity_pattern']
        exclued_pattern = initial.USER_PATTERN['exclued_pattern']
        
        if (self.userId 
            and re.search(digit_pattern, self.message) 
            and re.search(entity_pattern, self.message)
            and not re.search(exclued_pattern, self.message)
        ):
            self.message += f' and where customer/user id is {self.userId}'
            self.is_userId_attached = True
        
        return self.message

    async def __attach_pre_prompt(self):
        self.conversation_history = await self.memory.get_conversation()
        return initial.PRE_PROMPTS['system'].format(
            user_query = self.message,
            history = self.conversation_history
        )

    def __handle_greetings(self):
        greeting_pattern = initial.GREETING_PATTERNS['greeting_pattern']
        strict_gretting_pattern = initial.GREETING_PATTERNS['strict_gretting_pattern']
        
        response = None
        if (re.search(greeting_pattern, self.message)
            or re.search(strict_gretting_pattern, self.message)
        ):
            response = "👋 Hello! 😊 How can I help you today? 🤔"
        elif 'nice to meet you' in self.message:
            response = "🙂 Nice to meet you too! 👋 How can I assist you today? 🤔"
        elif 'how are you' in self.message:
            response = "😊 I'm good, thanks! 👍 Ready to help 🤔"

        if response:
            asyncio.create_task(self.memory.add_memory(self.message, response))
            
        return response
    
    def __entity_extractor(self, division: Literal['database', 'document', 'website']):
        # Extract named entities
        grammer_entities = ['NOUN', 'PROPN', 'ADJ']
        if division == "document":
            grammer_entities.append('VERB')
            
        doc = initial.NLP_PROCESSOR(self.input.message)
        entities = " ".join(list(set(ent.text for ent in doc if ent.pos_ in grammer_entities)))
        print("ENTTITIES...........", entities, division)
        return entities or None
    
    def __filter_tags(self):
        order_pattern = initial.FILTER_TAG_PATTERNS['order_pattern']
        product_pattern = initial.FILTER_TAG_PATTERNS['product_pattern']
        product_category_pattern = initial.FILTER_TAG_PATTERNS['product_category_pattern']
        category_pattern = initial.FILTER_TAG_PATTERNS['category_pattern']
        continue_pattern = initial.FILTER_TAG_PATTERNS['continue_pattern']
        
        redis_tag_key = f"{self.input.session_id}_tag_key"
        tag = None
        if (
            (re.search(product_category_pattern, self.input.message, re.IGNORECASE) 
            and re.search(product_pattern, self.input.message, re.IGNORECASE))
            or re.search(category_pattern, self.input.message, re.IGNORECASE)
        ):
            tag = 'category_tag'
        elif re.search(product_pattern, self.input.message, re.IGNORECASE):
            tag = 'product_tag'
        elif re.search(order_pattern, self.input.message, re.IGNORECASE):
            tag = 'order_tag'
        elif re.search(continue_pattern, self.input.message, re.IGNORECASE):
            if extracted_tag := initial.REDIS_CLIENT.get(redis_tag_key):
                tag = extracted_tag
        
        if tag:
            initial.REDIS_CLIENT.set(redis_tag_key, tag)
        return tag
    
    async def __find_division_type(self) -> Literal['website', 'database', 'document']:
        system_prompt = initial.PRE_PROMPTS['division'].format(
            user_query = self.message,
        )
        
        llm = initial.BASE_MODEL
        response = llm.predict_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.message)
        ])
        content = response.content
        print("DIVISION RESPONSE.........", content, response)
        if any(division for division in ['website', 'database', 'document'] if division in content):
            return content
        
        return 'database'
    
    