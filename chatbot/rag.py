import json
import re
import traceback
import asyncio
from typing import List, Literal, Optional
from langchain_chroma import Chroma
from rapidfuzz import fuzz, process as FuzzProcess

from chatbot.api_loader import ApiLoader
from chatbot.memory import CustomChatMemory
from chatbot.models import ChatInput
from langchain.chains import RetrievalQA
from langchain.schema import Document, BaseRetriever, SystemMessage, HumanMessage
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
    first_limit:int = 500
    second_limit:int = 50
    third_limit:int = 15
    empty_document = [
        Document(
            page_content = "No data found",
            metadata = {}
        )
    ]
    

    def __init__(self, input:ChatInput) -> None:
        self.input:ChatInput = input
        self.message = input.message.lower()
        self.platform = initial.PLATFORM_NAME
        self.userId = 1 #{'wordpress':1, 'mysql':14, 'sqlite':2}
        self.base_url = 'http://localhost:10003/'
        
        # Initialise memory
        self.memory:CustomChatMemory = CustomChatMemory()
        self.memory.add_user(self.input.session_id)
     
    async def invoke(self):
        try:           
            # Handle greeting prompts
            greeting_response = self.__handle_greetings()
            if greeting_response:
                return greeting_response
            
            # Attaching user id and preprompt
            self.__attach_userId()
            self.pre_prompt_message = await self.__attach_pre_prompt()
            
            response = await self._handle_cart_enquiry()
            if not response:
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

    async def _handle_cart_enquiry(self):
        if self.platform not in ['wordpress', 'shopify']:
            return None 
        
        if not self.is_userId_attached:
            return None

        tag = self.__filter_tags()
        if tag != 'cart_tag':
            return None
        
        loader = ApiLoader(self.platform, self.base_url)
        response = await loader._call_wp_api(
                                    'cart', 
                                    {'customer':self.userId},
                                    populate=False
                                )
        print("CART RESPONS........", response)
        if response:
            document = [Document(page_content=json.dumps(response))]
        else:
            document = self.empty_document

        retriever = DummyRetriever(filtered_docs=document)
        result = await self._retrieve_response(retriever)
        return result
        
    async def _retrieve_response(self, retriever = None):

        response_retriever = retriever or (await self._smart_retriever())
     
        # Create a RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=initial.BASE_MODEL, 
            chain_type="stuff", 
            retriever=response_retriever,
            return_source_documents=True,
        )
        
        invoked_response = qa_chain.invoke({'query':self.pre_prompt_message})
        response = invoked_response.get('result', "No response available.")
        return response          

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
            print("DIVISION..................", division)
            
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
        self.__limit_setter('first', set_limit=len(relevant_docs))

        filtered_documents = self._filter_documents(division, relevant_docs)
        self.__limit_setter('second', set_limit=len(filtered_documents))    
        
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
        search_kwargs= {'k':self.first_limit}

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
        print("USER ATTACHED.............", self.is_userId_attached)

        relevant_docs = documents
        filtered_content = []
        for doc in relevant_docs:
            content = doc.page_content
            if division == "database" and (self.is_userId_attached and self.userId):
                if not re.search(rf"\s{self.userId}[,]", doc.page_content):
                    continue
            filtered_content.append(content)

        filtered_docs: list[Document] = []
        filtered_content = self._smart_word_filter(division, filtered_content)
        if filtered_content:
            for content in filtered_content:     
                filtered_docs.append(Document(
                    page_content=content,
                    metadata={}
                ))

        print("FILTERED CONTENT........", filtered_content)
        # send filtered docs
        if filtered_docs:
            return filtered_docs
    
        # send blank docs
        return self.empty_document
    
    async def _re_filter_retriever(self, merged_docs):
        temp_vector_store = Chroma.from_documents(merged_docs, initial.EMBEDDING_FUNCTION)  
        temp_retriever = temp_vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs= {'k':self.third_limit}
        )
        print("LIMITS.................", self.first_limit, self.second_limit, self.third_limit)
        refined_results = temp_retriever.invoke(self.message)
        retriever = DummyRetriever(filtered_docs=refined_results)
        return retriever
        
    def _smart_word_filter(self, division: Literal['database', 'document', 'website'], content_list):
        query = self.input.message
        exclude_filter_tags = ['product_category_tag', 'post_category_tag']
        if division == 'database':
            if (
                (self.is_userId_attached and self.userId)
                or
                (self.filter_tag in exclude_filter_tags)
            ):
                return content_list
            
            entities = self.__entity_extractor() or query
            print("ENTITIES...............", entities)

        if content_list:
            # FuzzProcess.extract_iter **use it when you have large dataset**
            matches = FuzzProcess.extract(
                query, 
                content_list, 
                limit=self.second_limit,
                scorer=fuzz.partial_ratio, 
                score_cutoff=initial.FILTERING_MINIMUM_SCORE[division], 
                score_hint=70
            )
            print("MATCHES..................", matches)
            return list(map(lambda x: x[0], matches))
        
        return content_list

    
        
    def __attach_userId(self):
        
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
            company=initial.COLLECTION_NAME,
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

    def __entity_extractor(self):
        # Extract named entities
        grammer_entities = ['NOUN', 'PROPN', 'ADJ']          
        doc = initial.NLP_PROCESSOR(self.input.message)
        entities = " ".join(list(set(ent.text for ent in doc if ent.pos_ in grammer_entities)))
        return entities or []
    
    def __filter_tags(self):
        cart_pattern = initial.FILTER_TAG_PATTERNS['cart_pattern']
        order_pattern = initial.FILTER_TAG_PATTERNS['order_pattern']
        product_pattern = initial.FILTER_TAG_PATTERNS['product_pattern']
        post_pattern = initial.FILTER_TAG_PATTERNS['post_pattern']
        continue_pattern = initial.FILTER_TAG_PATTERNS['continue_pattern']
        helper_category_pattern = initial.FILTER_TAG_PATTERNS['helper_category_pattern']
        
        redis_tag_key = f"{self.input.session_id}_tag_key"
        tag = None
        
        if re.search(cart_pattern, self.input.message, re.IGNORECASE):
            tag = 'cart_tag'
            
        elif (
            re.search(helper_category_pattern, self.input.message, re.IGNORECASE) 
            and re.search(product_pattern, self.input.message, re.IGNORECASE)
        ):
            tag = 'product_category_tag'
        elif re.search(product_pattern, self.input.message, re.IGNORECASE):
            tag = 'product_tag'
            
        elif re.search(order_pattern, self.input.message, re.IGNORECASE):
            tag = 'order_tag'
            
        elif (
            re.search(helper_category_pattern, self.input.message, re.IGNORECASE) 
            and re.search(post_pattern, self.input.message, re.IGNORECASE)
        ):
            tag = 'post_category_tag'
        elif re.search(post_pattern, self.input.message, re.IGNORECASE):
            tag = 'post_tag'
            
        elif re.search(continue_pattern, self.input.message, re.IGNORECASE):
            if extracted_tag := initial.REDIS_CLIENT.get(redis_tag_key):
                tag = extracted_tag
        
        if tag:
            initial.REDIS_CLIENT.set(redis_tag_key, tag)

        self.filter_tag = tag
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
        if any(division for division in ['website', 'database', 'document'] if division in content):
            return content
        
        return 'database'
    
    def __limit_setter(self, level: Literal['first', 'second', 'third'], set_limit: int):
        setattr(self, f"{level}_limit", set_limit)

        # Adjust second limit based on first limit
        if self.first_limit > 1200:
            self.second_limit = 120
        elif self.first_limit > 600:
            self.second_limit = 80
        elif self.first_limit > 300:
            self.second_limit = 50
        elif self.first_limit > 100:
            self.second_limit = 30
        elif self.first_limit > 50:
            self.second_limit = 25
        elif self.first_limit > 25:
            self.second_limit = 15
        else:
            self.second_limit = 10

        # Adjust third limit based on second limit
        if self.second_limit > 100:
            self.third_limit = 30
        elif self.second_limit > 50:
            self.third_limit = 20
        elif self.second_limit > 30:
            self.third_limit = 15
        elif self.second_limit > 20:
            self.third_limit = 12
        elif self.second_limit > 14:
            self.third_limit = 8
        else:
            self.third_limit = 5

        

            

