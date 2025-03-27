import asyncio
import json
import re
import traceback
import pandas as pd
from typing import List
from sqlalchemy import create_engine, inspect
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chatbot.api_loader import ApiLoader
from chatbot.extractor import ContentExtractor
from PyPDF2 import PdfReader
import initial


class ChromaDBPopulator:
    
    def __init__(self):
        """Initialize ChromaDB Populator."""
        
        self.config = initial.GET_CONFIGS('database')
                  
    def _connect_to_database(self):
        """Establish a connection to the database dynamically based on config."""
        
        db_type = self.config.get("type")
        username = self.config.get("username", "")
        password = self.config.get("password", "")
        host = self.config.get("host", "")
        port = self.config.get("port", "")
        database = self.config.get("database", "")

        if db_type == "sqlite":
            db_url = f"sqlite:///{database}"
        elif db_type == "postgresql":
            db_url = f"postgresql://{username}:{password}@{host}:{port}/{database}"
        elif db_type == "mysql":
            db_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        try:
            engine = create_engine(db_url)
            print(f"✅ Successfully connected to {db_type} database!")
            return engine
        except Exception as e:
            raise ConnectionError(f"❌ Database connection failed: {e}")
       
    async def populate_chroma_db(self):
        """
            Main function to populate ChromaDB with following sources
                1. structured database
                2. documents (.json, .txt, .pdf) 
                3. website page content (Normal - {selenium, bs4}, wordpress)
        """

        try:
            # Load Tables
            # await self.load_tables_data()

            # Load Documents
            # await self.load_documents_data()
            
            # Load website content
            # await self.load_websites_content_data()

            # Load Apis data
            # await self.load_apis_data()
            pass

        except Exception as e:
            traceback.print_exc()
            print(f"❌ Chroma Populator error: {e}")

        print("🚀 ChromaDB population complete!")
        

    # Document loader
    async def load_documents_data(self):
        """Load data from '.json' file and store in ChromaDB."""
        
        vectorstore = initial.VECTOR_DB['document'](initial.COLLECTION_NAME)
        for document in self.config['documents']:
            file_extension = document.split('.')[-1].lower()
            
            if file_extension == 'json':  # Check if the document is a .json file
                await self.__load_json_document(document, vectorstore)
            elif file_extension == 'txt':  # Check if the document is a .txt file
                await self.__load_txt_document(document, vectorstore)
            elif file_extension in ['pdf']:  # Check if the document is a .pdf file
                await self.__load_pdf_document(document, vectorstore)
            else:
                print(f"Document type not defined - {document}!")
                
    async def __load_json_document(self, document, vectorstore):
        # Able to load json type files, with QA format

        try:
            with open(document, "r") as file:
                data = json.load(file)

            # Modify docs to include priority metadata
            documents = [
                Document(
                    page_content=f"Question: {qa.get('question')}\nAnswer: {qa.get('answer')}", 
                    metadata={"devision": initial.DIVISIONS["doc"], "source": document}
                ) 
                for qa in data
            ]

            if documents:
                await vectorstore.aadd_documents(documents)
                print(f"📂 Added {len(documents)} low-priority documents from '{document}'")

        except Exception as e:
            print(f"Error loading document: {document} - {e}")
        
    async def __load_txt_document(self, document, vectorstore):
        try:
            with open(document, "r") as file:
                text = file.read()

            # Apply overlapping chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(text)
            
            documents = [
                Document(
                    page_content=chunk,
                    metadata={"devision": initial.DIVISIONS["doc"], "source": document}
                )
                for chunk in chunks
            ]
            
            if documents:
                await vectorstore.aadd_documents(documents)
                print(f"📂 Added {len(documents)} low-priority documents from '{document}'")
        except Exception as e:
            print(f"Error loading document: {document} - {e}")

    async def __load_pdf_document(self, document, vectorstore):
        # Method to load .pdf documents
        try:
            # Using PyMuPDF to extract text from PDF
            reader = PdfReader(document)
            text = ""

            for page in reader.pages:
                text += page.extract_text() or ""  # Extract text safely
        
            if not text.strip():
                print(f"⚠️ No text extracted from '{document}'! The PDF might be image-based.")

            # Apply overlapping chunking
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_text(text)

            documents = [
                Document(
                    page_content=chunk,
                    metadata={"devision": initial.DIVISIONS["doc"], "source": document}
                )
                for chunk in chunks
            ]

            if documents:
                await vectorstore.aadd_documents(documents)
                print(f"📂 Added {len(documents)} low-priority documents from '{document}'")
        except Exception as e:
            print(f"Error loading document: {document} - {e}")
   
    
    # Database Table handlers     
    async def load_tables_data(self):
        
        # Connecting with database and starting an engine
        self.engine = self._connect_to_database()
        self.inspector = inspect(self.engine)
        # self.inspector = inspect(SQLITE_DATABASE_ENGINE)
        
        table_names = self.inspector.get_table_names()
        vectorstore = initial.VECTOR_DB['database'](initial.COLLECTION_NAME)

        for table in self.config.get('relational_tables'):
            if table not in table_names:
                print(f"⚠️ Table '{table}' does not exist. Skipping...")
                continue
            await self._process_related_table(vectorstore, table)
            
        for table in self.config.get('independent_tables'):
            if table not in table_names:
                print(f"⚠️ Table '{table}' does not exist. Skipping...")
                continue
            await self._process_independent_table(vectorstore, table)

        print("Database extraction Done!")
      
    async def _process_independent_table(self, vectorstore, table):
        """Process tables that do not have foreign keys."""
        
        try:
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                print(f"⚠️ Table '{table}' has no data. Skipping...")
                return

            df["text"] = df.apply(lambda row: ", ".join( f"{table}_id" if col == "id" else col + f": {row[col]}" for col in df.columns), axis=1)

            tag = self.__filter_tag(table)
            documents = [
                Document(
                    page_content=row["text"],
                    metadata={
                        "devision": initial.DIVISIONS["db"], 
                        "source": table, 
                        "tags":tag
                    }
                )
                for _, row in df.iterrows()
            ]

            if documents:
                await self.__paadd_documents(documents, vectorstore)
                print(f"✅ Added {len(documents)} records from '{table}' into ChromaDB!")
        except Exception as e:
            print(f"Unable to populate add {table} table: {str(e)}")
            traceback.print_exc()

    async def _process_related_table(self, vectorstore, table):
        """Process tables with hierarchical relationships dynamically based on the config."""
        
        relational_config = self.config.get('relational_tables', {})
        if table not in relational_config:
            return

        relations = relational_config[table]
        join_clauses = []
        select_columns = []

        # Define alias for the base table to prevent conflicts
        table_alias = table

        # Get column names for the base table
        base_columns = [col["name"] for col in self.inspector.get_columns(table)]
        select_columns.extend([f"{table_alias}.{col} AS {table_alias}_{col}" for col in base_columns])

        # Iterate over related tables
        def process_child_tables(current_table, current_alias, child_relations):
            for ref_table, details in child_relations.items():
                ref_alias = f"{current_alias}_{ref_table}"  # Alias to prevent conflicts

                # Extract foreign key information
                fk_col = details.get("fk")
                if not fk_col:
                    print(f"⚠️ Skipping table '{ref_table}' due to missing foreign key mapping.")
                    continue

                if ref_table in self.inspector.get_table_names():
                    # Get primary key of the related table
                    pk_info = self.inspector.get_pk_constraint(ref_table)
                    ref_primary_key = pk_info["constrained_columns"][0] if pk_info and "constrained_columns" in pk_info else None
                    
                    if ref_primary_key:
                        join_clauses.append(f"LEFT JOIN {ref_table} AS {ref_alias} ON {current_alias}.{fk_col} = {ref_alias}.{ref_primary_key}")

                        # Get column names for the reference table
                        ref_columns = [col["name"] for col in self.inspector.get_columns(ref_table)]

                        # Rename columns to prevent conflicts
                        select_columns.extend([f"{ref_alias}.{col} AS {ref_alias}_{col}" for col in ref_columns])
                        
                        # Recursively process child relationships if defined
                        if "child" in details:
                            process_child_tables(ref_table, ref_alias, details["child"])
                    else:
                        print(f"⚠️ Could not determine primary key for table '{ref_table}'. Skipping...")

        # Process first-level child tables
        process_child_tables(table, table_alias, relations["child"])

        # Build the final SQL query
        query = f"SELECT {', '.join(select_columns)} FROM {table} AS {table_alias} {' '.join(join_clauses)}"

        # Fetch the data
        merged_df = pd.read_sql(query, self.engine)
        
        if merged_df.empty:
            print(f"⚠️ No related data found for '{table}'. Skipping...")
            return
        
        merged_df["text"] = merged_df.apply(
            lambda row: ", ".join(
                f"{' '.join(c for c in col.split('_'))}: {str(row[col])}"  # Ensure proper formatting
                for col in merged_df.columns
            ), 
            axis=1
        )
        
        tag = self.__filter_tag(table)
        documents = [
            Document(
                page_content=row["text"],
                    metadata={
                        "devision": initial.DIVISIONS["db"], 
                        "source": table, 
                        "tags":tag
                    }
            )
            for _, row in merged_df.iterrows()
        ]

        if documents:
            await self.__paadd_documents(documents, vectorstore)
            print(f"✅ Added {len(documents)} merged records from '{table}' into ChromaDB!")
         
       
    # Website content loader 
    async def load_websites_content_data(self):     
        try:
            vectorstore = initial.VECTOR_DB['website'](initial.COLLECTION_NAME)
            web_url:str = self.config['website_url']
            
            extractor = ContentExtractor()
            links = extractor.extract_links(web_url)
            content = extractor.extract_content(links)
            
            if len(content.keys()) < 1:
                print(f"No content extracted from website url: {web_url}")
                
            documents = []
            for url, data in content.items():
                if data:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
                    chunks = splitter.split_text(data)
                    
                    documents.extend([
                        Document(
                            page_content=chunk,
                            metadata={"devision": initial.DIVISIONS["web"], "source": url}
                        )
                        for chunk in chunks
                    ])

            if documents:
                await self.__paadd_documents(documents, vectorstore)
                print(f"📂 Added {len(documents)} website content from following urls: {content.keys()}")
        except Exception as e:
            traceback.print_exc()
            print(f"Error loading website content - {e}")


    # Apis data loader
    async def load_apis_data(self):
        base_url = 'http://localhost:10003/'
        vectorstore = initial.VECTOR_DB['database'](initial.COLLECTION_NAME)

        loader = ApiLoader(base_url, vectorstore)
        await loader.load("wordpress", ['post_category', 'posts'])
        await loader.load("woocommerce", ['product_category', 'products', 'orders'])
        # cart_data = await api_loader._call_wp_api('cart', {'customer':"1"})
        
    
    # Helper functions
    async def __paadd_documents(self, documents:List[Document], vectorstore):
        try:
            print("Starting adding documents parallely .........")
            
            num_threads = 5
            chunk_size = max(1, len(documents) // num_threads)  # Avoid division by zero
            document_chunks = [documents[i:i + chunk_size] for i in range(0, len(documents), chunk_size)]

            async def populate_documents(vectorstore, document_chunks):
                """Populate vectorstore with documents asynchronously."""
                tasks = [vectorstore.aadd_documents(document) for document in document_chunks]
                await asyncio.gather(*tasks)  # Runs all coroutines concurrently

            # Calling all the 5 threads parallely
            asyncio.run(populate_documents(vectorstore, document_chunks))
            return True
        except Exception as e:
            print(f"Error while adding documents - {str(e)}")
            return None
    
    def __chunk_text(self, text, chunk_size=500, overlap_size=100):
        """Chunk text with overlapping chunks."""
        
        chunks = []
        for start in range(0, len(text), chunk_size - overlap_size):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
        return chunks
        
    def __filter_tag(self, tablename:str):
        order_pattern = initial.CHROMA_FILTER_PATTERNS['order_pattern']
        product_pattern = initial.CHROMA_FILTER_PATTERNS['product_pattern']
        product_category_pattern = initial.CHROMA_FILTER_PATTERNS['product_category_pattern']

        tag = 'None'
        if re.search(order_pattern, tablename, re.IGNORECASE):
            tag = 'order_tag'
        elif re.search(product_pattern, tablename, re.IGNORECASE):
            tag = 'product_tag'
        elif re.search(product_category_pattern, tablename, re.IGNORECASE):
            tag = 'product_category_tag'
        
        return tag

