import os
import re
import shutil
import time
import traceback
from typing import Literal
from sqlalchemy import create_engine, inspect, text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from langchain_core.documents import Document
import json
import spacy
import toml

nlp = spacy.load("en_core_web_sm")

def get_configs(key:str):
    try:
        with open("config.json", "r") as file:
            config = json.load(file)
            return config.get(key)
    except FileNotFoundError:
        raise ValueError(" Config file not found!")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in config file!")

def vector_store():
    config = get_configs('database')
    persist_directory= config.get('vector_db_directory')
    embedding_function = HuggingFaceEmbeddings()
    vectorstore = Chroma(persist_directory='chatbot/chroma_directory', embedding_function=embedding_function)
    return vectorstore

class ChromaDBPopulator:
    
    def __init__(self):
        """Initialize ChromaDB Populator."""
        
        self.config = get_configs('database')
        # if self.config.get('vector_db_reset'):
        #     self._reset_db()
            
        self.engine = self._connect_to_database()
        self.inspector = inspect(self.engine)
        self.vectorstore = vector_store()
        # self.inspector = inspect(SQLITE_DATABASE_ENGINE)
        
    def _reset_db(self):
        self.persist_directory = self.config.get('vector_db_directory')
        if os.path.exists(self.persist_directory):
            print("Removing existing ChromaDB directory...")
            shutil.rmtree(self.persist_directory)
            time.sleep(4)
            
            os.makedirs(self.persist_directory, exist_ok=True)
            self.vectorstore = vector_store()
        else:
            print("Unable to reset ChromaDB!",self.persist_directory, os.path.exists(self.persist_directory))
            
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

    def _load_faq_data(self):
        """Load FAQ data from 'faq.txt' and store in ChromaDB."""
        
        for document in self.config['documents']:
            if document.split('.')[-1] == 'txt':  # Check if the document is a .txt file
                try:
                    loader = TextLoader(document)  # Load the correct document file dynamically
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    docs = text_splitter.split_documents(documents)

                    # Modify docs to include priority metadata
                    documents = [
                        Document(
                            page_content=doc.page_content, 
                            metadata={
                                "content": doc.page_content,  # Ensure this is a string
                                "priority": "low",
                                "type": self._extract_keywords(doc.page_content),
                                "source": doc.metadata.get("source", "")  # Ensure metadata values are strings
                            }
                        ) 
                        for doc in docs
                    ]
                    
                    if documents:
                        self.vectorstore.add_documents(documents)
                        print(f"📂 Added {len(documents)} low-priority documents from '{document}'")

                except Exception as e:
                    print(f"Error loading document: {document} - {e}")
            else:
                print(f"Document type not defined - {document}!")
                
    def _process_independent_table(self, table):
        """Process tables that do not have foreign keys."""
        
        try:
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                print(f"⚠️ Table '{table}' has no data. Skipping...")
                return

            df["text"] = df.apply(lambda row: ", ".join( f"{table}_col" if col == "id" else col + f": {row[col]}" for col in df.columns), axis=1)

            documents = [
                Document(
                    page_content=row["text"],
                    metadata={col: str(row[col]) for col in df.columns} | {"priority": "high", "type": self._extract_keywords(" ".join(df.columns))}
                )
                for _, row in df.iterrows()
            ]
            print("DOCUMENTS............", documents)
            if documents:
                self.vectorstore.add_documents(documents)
                print(f"✅ Added {len(documents)} records from '{table}' into ChromaDB!")
        except Exception as e:
            print(f"Unable to populate add {table} table: {str(e)}")
            traceback.print_exc()

    def __process_related_table(self, table):
        """Process tables with manually defined relationships from the config file."""
        
        relational_config = self.config.get('relational_tables')
        if table not in relational_config:
            return

        relations = relational_config[table]
        join_clauses = []
        select_columns = []

        # Get column names for the base table
        base_columns = [col["name"] for col in self.inspector.get_columns(table)]
        select_columns.extend([f"{table}.{col} AS {table}_{col}" for col in base_columns])

        # Iterate over related tables
        for ref_table, fk_col in relations.items():
            if ref_table in self.inspector.get_table_names():
                # Get the actual primary key of the related table
                pk_info = self.inspector.get_pk_constraint(ref_table)
                ref_primary_key = pk_info["constrained_columns"][0] if pk_info and "constrained_columns" in pk_info else None
                
                if ref_primary_key:
                    join_clauses.append(f"LEFT JOIN {ref_table} ON {table}.{fk_col} = {ref_table}.{ref_primary_key}")

                    # Get column names for the reference table
                    ref_columns = [col["name"] for col in self.inspector.get_columns(ref_table)]

                    # Rename columns to prevent conflicts
                    select_columns.extend([f"{ref_table}.{col} AS {ref_table}_{col}" for col in ref_columns])
                else:
                    print(f"Warning: Could not determine primary key for table {ref_table}")

        # Build the final SQL query
        query = f"SELECT {', '.join(select_columns)} FROM {table} {' '.join(join_clauses)}"
        
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
        
        documents = [
            Document(
                page_content=row["text"],
                metadata={col: str(row[col]) for col in merged_df.columns}  | {"priority": "high"}
            )
            for _, row in merged_df.iterrows()
        ]

        if documents:
            self.vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} merged records from '{table}' into ChromaDB!")


    def _process_related_table(self, table, alias=None, parent_table=None, parent_fk=None):
        """Process tables with hierarchical relationships dynamically based on the config."""
        
        relational_config = self.config.get('relational_tables', {})
        if table not in relational_config:
            return

        relations = relational_config[table]
        join_clauses = []
        select_columns = []

        # Define alias for the base table to prevent conflicts
        table_alias = alias or table

        # Get column names for the base table
        base_columns = [col["name"] for col in self.inspector.get_columns(table)]
        select_columns.extend([f"{table_alias}.{col} AS {table_alias}_{col}" for col in base_columns])

        # Handle parent table relationship if applicable
        if parent_table and parent_fk:
            join_clauses.append(f"LEFT JOIN {table} AS {table_alias} ON {parent_table}.{parent_fk} = {table_alias}.{parent_fk}")

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
        process_child_tables(table, table_alias, relations)

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
        
        documents = [
            Document(
                page_content=row["text"],
                metadata={col: str(row[col]) for col in merged_df.columns} | {"priority": "high", "type": self._extract_keywords(" ".join(merged_df.columns))}
            )
            for _, row in merged_df.iterrows()
        ]
        print("DOCUMENTS............", documents)
        if documents:
            self.vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} merged records from '{table}' into ChromaDB!")
            
    def populate_chroma_db(self):
        """Main function to populate ChromaDB with structured database data & FAQ documents."""

        try:
            table_names = self.inspector.get_table_names()

            for table in self.config.get('relational_tables'):
                if table not in table_names:
                    print(f"⚠️ Table '{table}' does not exist. Skipping...")
                    continue
                self._process_related_table(table)
                
            for table in self.config.get('independent_tables'):
                if table not in table_names:
                    print(f"⚠️ Table '{table}' does not exist. Skipping...")
                    continue
                self._process_independent_table(table)

            # Load FAQ data after database population
            self._load_faq_data()

        except Exception as e:
            traceback.print_exc()
            print(f"❌ Database error: {e}")

        print("🚀 ChromaDB population complete!")
        
    def _extract_keywords(self, text):
        text = text.replace("_", " ")
        doc = nlp(text)
        
        # Extract first PROPN, NOUN, or ADJ keyword
        keywords = []
        for token in doc:
            if token.pos_ in {"PROPN", "NOUN"}:
                keywords.append(token.text) 

        # If no keyword found, use regex to find predefined keywords
        matches = re.findall(r'\b(order|orders|customer|customers|shipping|product|products)\b', text, re.IGNORECASE)
        if matches:
            keywords.extend(matches)
            
        keywords = list(set(keywords))
        return ",".join(keywords)


# Load the existing vector database
vectorstore = vector_store()


def extract_dynamic_filters(user_input):
    """
    Extracts relevant keywords dynamically from the user query.
    Supports multiple filters like email, name, id, price, and category.
    """
    doc = nlp(user_input)
    extracted_values = {}
    
    price_pattern = r'\b(?:price|under|between|cheaper than)\b'
    name_pattern = r'\b(?:user|i am|my name|name|category|product|product name|available)\b'

    for token in doc:
        print("x............", token, token.pos_, token.pos_, token.like_url)
        if token.like_email:
            extracted_values["email"] = token.text  # Capture email
            print("email.............", extracted_values)
            break
        elif token.like_num:
            if "id" in user_input.lower():
                extracted_values["id"] = int(token.text)  # Capture ID
                print("id.............", extracted_values)
                break
            elif re.search(price_pattern, user_input.lower(), re.IGNORECASE):
                extracted_values["price"] = float(token.text)  
                print("price.............", extracted_values)
                break
        elif token.pos_ in ["PROPN", "NOUN"]:  
            if re.search(name_pattern, user_input.lower(), re.IGNORECASE):
                key = "category" if "category" in user_input.lower() else "name"
                extracted_values[key] = token.text  
                print("last.............", extracted_values)
                break
    return extracted_values

def retrieve_response(prompt, message, llm, priority:Literal['high', 'low']):
    search_kwargs = {
        'k':15 if priority == 'high' else 5,
        'filter': {
            '$and':[
                {'priority': priority}
            ]
        }
    }
    match = re.search(r'\b(order|orders|customer|customers|shipping|product|products)\b', message, re.IGNORECASE)
    if match:
        search_kwargs['filter']['$and'].append({'type': match.group()})
    print("SEARCH KWARES..............", search_kwargs)
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    # Create a RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        return_source_documents=True,
    )
    invoked_response = qa_chain.invoke({'query':prompt})
    print("INVOKED RESPONSE...............", invoked_response)
    response = invoked_response.get('result', "No response available.")
    return response

