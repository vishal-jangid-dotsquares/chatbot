import json
import re
import traceback
import asyncio
from typing import Any, List, Literal, Optional

from langchain_chroma import Chroma
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.schema import Document, BaseRetriever, SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
from core.api_loader import PLATFORM_TYPES, ApiLoader
from core.memory import CustomChatMemory
from core.models import ChatInput
import initial
from langchain_core.runnables import RunnableLambda as runnable, RunnablePassthrough, RunnableParallel, RunnableBranch, RunnableSequence



DIVISION_TYPE = Literal['database', 'document', 'website']
    
class DummyRetriever(BaseRetriever):
    filtered_docs: list[Document]

    def _invoke(self, query: str) -> list[Document]:
        return self.filtered_docs

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.filtered_docs
    

class Rag:
    userId:Optional[int] = None 
    is_userId_attached:Optional[bool] = False
    filter_tag:Optional[str] = None
    is_followUp:bool = False
    pre_prompt_message:str = ""
    first_limit:int = 400
    second_limit:int = 50
    empty_document = [
        Document(
            page_content = "No data found",
            metadata = {}
        )
    ]
    

    def __init__(self, input:ChatInput) -> None:
        self.input:ChatInput = input
        self.input.message = input.message.strip().lower()
        self.message = self.input.message
        
        self.platform:PLATFORM_TYPES = initial.PLATFORM_NAME
        self.userId = 26 #{'wordpress':1, 'woocommerce':20, 'mysql':20, 'sqlite':2}
        self.base_url = 'http://localhost:10003/'
        
        # Initialise memory
        self.memory: CustomChatMemory = CustomChatMemory(self.input.session_id)
     
     
    async def invoke(self) -> str:
        try:           
            # Attaching user id
            self.__attach_userId()
            response = await self._response_pipeline()

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

    async def _response_pipeline(self) -> str:
        # Handle greeting prompts
        if greeting_response := self.__greeting_handler():
            return greeting_response

        # detect follow up prompts
        self.is_followUp = await self.__detect_follow_up_query()
        print("IS FOLLLOW UP..........", self.is_followUp)
        print("IS USER ATTACHED...............", self.is_userId_attached, self.userId)
        
        # Handle cart related prompts
        cart_retriever = await self._handle_cart_enquiry()
        if cart_retriever:
            division, retriever = cart_retriever
        else:
            division, retriever = await self._smart_retriever()
        
        # refine the retriever
        if division is not None:
            retriever = await self._refine_retriever(division, retriever)

        response = await self._retrieve_response(retriever=retriever)
        return response
        
    async def _retrieve_response(self, retriever) -> str:
        await self.__attach_pre_prompt()

        # Create a RetrievalQA Chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=initial.MODELS['vision'], 
            chain_type="stuff", 
            retriever=retriever,
            return_source_documents=True,
        )
        
        invoked_response = qa_chain.invoke({'query':self.pre_prompt_message})
        response = invoked_response.get('result', "No response available.")
        return response  

    async def _handle_cart_enquiry(self):
        if self.platform not in ['wordpress', 'shopify']:
            return None 
        
        if not self.is_userId_attached:
            return None

        tag = await self.__filter_tags()
        if tag != 'cart_tag':
            return None
        
        loader = ApiLoader(self.platform, self.base_url)
        response = await loader._call_wp_api(
                                    'cart', 
                                    {'customer':self.userId},
                                    populate=False
                                )

        if response:
            document = [Document(page_content=json.dumps(response))]
        else:
            document = self.empty_document

        retriever = DummyRetriever(filtered_docs=document)
        return 'database', retriever
        
    async def _smart_retriever(self):
        # it retriever parallely
        # first it retireve devision type
        # and then it retrive all the documents parallely
        if self.is_followUp:
            division = await self.memory.get_last_division()
            last_question = await self.memory.get_last_message() or self.message
            if division and last_question:
                self.message = last_question.strip()
                print("IS FOLLOW............", division, self.message)
                retriever = await self._get_retriever(division)
            else:
                retriever = self.__get_fallback_retriever()
                division = None

        elif self.is_userId_attached:
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
        
        await self.memory.add_division(division or '')
        return division, retriever

    async def _refine_retriever(self, division:DIVISION_TYPE, retriever) -> DummyRetriever:   
        if division != "database":
            return retriever
        
        if not (self.is_userId_attached and self.userId):
            return retriever

        relevant_docs = retriever.invoke(input=self.message) 
        self.__limit_setter('first', set_limit=len(relevant_docs))
        print("FIRST FILTER......................\n", relevant_docs)

        # User ID base filtering
        filtered_docs: list[Document] = []

        userId_key_value_pattern = rf"(?:consumers?|customers?|users?|buyers?)(?:s|_id|id)?\s*:\s*\b{self.userId}\b"
        compiled_pattern = re.compile(userId_key_value_pattern, re.IGNORECASE)
        for doc in relevant_docs:
            if compiled_pattern.search(doc.page_content):
                filtered_docs.append(doc)

        print("FILTERED CONTENT........", filtered_docs)

        if filtered_docs:
            retriever = DummyRetriever(filtered_docs=relevant_docs)
            return retriever
        else:
            return self.__get_fallback_retriever()

    async def _retrieve_documents_parallely(self):
        results = await asyncio.gather(
            self._get_retriever("database"),
            self._get_retriever("document"),
            self._get_retriever("website")
        )
        return results

    async def _get_retriever(self, division: DIVISION_TYPE):
        vector_store = initial.VECTOR_DB[division](initial.COLLECTION_NAME)   
        self.first_limit = 200
        if division != 'database':
            self.first_limit = 100
            
        search_kwargs= {
            'k':10, 
            'fetch_k' : self.first_limit, 
            "lambda_mult": 0.8
        }

        # filtering with tags
        filter_tag = None
        if division == 'database':
            if filter_tag := (await self.__filter_tags()): 
                print("FILTER TAG................", filter_tag)
                search_kwargs['filter'] = {'tags': filter_tag}
            
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        
        return retriever
        
    async def _re_filter_documents(self, relevant_docs:List[Document]) -> List[Document]:
        # Return if empty docs
        if not relevant_docs:
            return self.empty_document
        
        docs_without_meta:List[Document] = [
            Document(
                page_content=doc.page_content,
                metadata={}
            ) for doc in relevant_docs
        ]
        print("DOC WITHOUT META.......................", docs_without_meta)
        temp_vector_store = Chroma.from_documents(docs_without_meta, initial.EMBEDDING_FUNCTION)  
        temp_retriever = temp_vector_store.as_retriever(
            search_type="mmr", 
            search_kwargs= {'k':self.second_limit, "lambda_mult": 0.8},
            persist_directory=None 
        )
        refined_results = temp_retriever.invoke(self.message)
        
        # Return if empty docs
        if not refined_results:
            return self.empty_document
        
        return refined_results
 
 
    def __attach_userId(self) -> str:
        
        self_reffering_pattern = initial.USER_PATTERN['self_reffering_pattern']
        entity_pattern = initial.USER_PATTERN['entity_pattern']
        exclued_pattern = initial.USER_PATTERN['exclued_pattern']
        
        if (self.userId 
            and re.search(self_reffering_pattern, self.message, re.IGNORECASE) 
            and re.search(entity_pattern, self.message, re.IGNORECASE)
            and not re.search(exclued_pattern, self.message, re.IGNORECASE)
        ):
            self.message += f' and where customer/user id = {self.userId}'
            self.is_userId_attached = True
        
        return self.message

    async def __attach_pre_prompt(self) -> str:
        
        if self.is_followUp:
            question = self.input.message
            last_question = self.message
        else:
            question = self.message
            last_question = await self.memory.get_last_message()

        conversation_history = await self.memory.get_conversation()
        self.pre_prompt_message = initial.PRE_PROMPTS['system'].format(
            company=initial.COLLECTION_NAME,
            current_question=question,
            last_question=last_question,
            history = conversation_history
        )
        return self.pre_prompt_message

    def __greeting_handler(self) -> Optional[str]:
        greeting_pattern = initial.GREETING_PATTERNS['greeting_pattern']
        strict_gretting_pattern = initial.GREETING_PATTERNS['strict_gretting_pattern']
        
        response = None
        if (re.search(greeting_pattern, self.message)
            or re.search(strict_gretting_pattern, self.message)
        ):
            response = "👋 Hello! How can I help you today? 😊"
        elif 'nice to meet you' in self.message:
            response = "🙂 Nice to meet you too! 👋 How can I assist you today? 🤔"
        elif 'how are you' in self.message:
            response = "I'm good, thanks! 👍 Ready to help 😊"

        if response:
            asyncio.create_task(self.memory.add_memory(self.message, response))
            
        return response

    async def __filter_tags(self) -> Optional[str]:
        cart_pattern = initial.FILTER_TAG_PATTERNS['cart_pattern']
        order_pattern = initial.FILTER_TAG_PATTERNS['order_pattern']
        product_pattern = initial.FILTER_TAG_PATTERNS['product_pattern']
        post_pattern = initial.FILTER_TAG_PATTERNS['post_pattern']
        helper_category_pattern = initial.FILTER_TAG_PATTERNS['helper_category_pattern']
        excluding_category_pattern = initial.FILTER_TAG_PATTERNS['excluding_category_pattern']
        
        tag = None
        # if its a follow up question just return previous filter tag
        if self.is_followUp:
            if tag := await self.memory.get_last_filter_tag():
                return tag or None

        # Check for category-based tags first to avoid conflicts
        if (
            helper_category_pattern.search(self.input.message) 
            and not excluding_category_pattern.search(self.input.message)
        ):
            if product_pattern.search(self.input.message):
                tag = 'product_category_tag'
            elif post_pattern.search(self.input.message):
                tag = 'post_category_tag'

        # Check for other individual tags
        if tag is None:
            if cart_pattern.search(self.input.message):
                tag = 'cart_tag'
            elif product_pattern.search(self.input.message):
                tag = 'product_tag'
            elif order_pattern.search(self.input.message):
                tag = 'order_tag'
            elif post_pattern.search(self.input.message):
                tag = 'post_tag'
        
        if tag:
            await self.memory.add_filter_tag(tag)

        self.filter_tag = tag
        return tag
    
    async def __find_division_type(self) -> DIVISION_TYPE:
        system_prompt = initial.PRE_PROMPTS['division'].format(
            user_query = self.message,
        )
        
        llm = initial.MODELS['vision']
        response:Any = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=self.message)
        ])

        content = response.content
        if any(division for division in ['website', 'database', 'document'] if division in content):
            return content
        
        return 'database'

    async def __detect_follow_up_query(self) -> bool:
        """Detects if the current query is a follow-up query."""
        # return any(pattern.search(self.input.message) for pattern in initial.FOLLOW_UP_PATTERN)
        
        prompt = initial.PRE_PROMPTS['followUp'].format(
            prev_query = await self.memory.get_last_message(),
            current_query = self.input.message
        )
    
        llm:BaseChatModel =  initial.MODELS['vision']
        response:Any = llm.invoke(
            [HumanMessage(content=prompt)]
        )
        print("DETECT FOLLOW UP......................", self.input.message, await self.memory.get_last_message())
        print("FOLLOW UP CONTENT.........................", response.content)
        answer = response.content.lower()
        is_follow_up = "yes" in answer
        return is_follow_up
        
    def __limit_setter(self, level: Literal['first', 'second'], set_limit: int):
        setattr(self, f"{level}_limit", set_limit)

        # Adjust second limit based on first limit
        if self.first_limit > 1200:
            self.second_limit = 30
        elif self.first_limit > 600:
            self.second_limit = 25
        elif self.first_limit > 300:
            self.second_limit = 20
        else:
            self.second_limit = 10
    
    def __get_fallback_retriever(self) -> DummyRetriever:
        retriever = DummyRetriever(filtered_docs = self.empty_document)
        return retriever
            

