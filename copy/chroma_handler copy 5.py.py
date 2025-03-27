import json
import re
import traceback
import pandas as pd
from sqlalchemy import create_engine, inspect
from langchain_core.documents import Document
from langchain.schema import BaseRetriever
import initial


class ChromaDBPopulator:
    
    def __init__(self):
        """Initialize ChromaDB Populator."""
        
        self.config = initial.GET_CONFIGS('database')
        self.engine = self._connect_to_database()
        self.inspector = inspect(self.engine)
        # self.inspector = inspect(SQLITE_DATABASE_ENGINE)
                  
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

    def _load_documents_data(self):
        """Load data from '.json' file and store in ChromaDB."""
        
        vectorstore = initial.VECTOR_DB['document'](initial.COLLECTION_NAME)
        for document in self.config['documents']:
            if document.split('.')[-1] == 'json':  # Check if the document is a .txt file
                try:
                    with open(document, "r") as file:
                        data = json.load(file)

                    # Modify docs to include priority metadata
                    documents = [
                        Document(
                            page_content=f"Question: {qa.get('question')}\nAnswer: {qa.get('answer')}", 
                            metadata={
                                "priority": "low",
                                "source": document
                            }
                        ) 
                        for qa in data
                    ]

                    if documents:
                        vectorstore.add_documents(documents)
                        print(f"📂 Added {len(documents)} low-priority documents from '{document}'")

                except Exception as e:
                    print(f"Error loading document: {document} - {e}")
            else:
                print(f"Document type not defined - {document}!")
                
    def _process_independent_table(self, vectorstore, table):
        """Process tables that do not have foreign keys."""
        
        try:
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, self.engine)

            if df.empty:
                print(f"⚠️ Table '{table}' has no data. Skipping...")
                return

            df["text"] = df.apply(lambda row: ", ".join( f"{table}_id" if col == "id" else col + f": {row[col]}" for col in df.columns), axis=1)

            tag = self.filter_tag(table)
            documents = [
                Document(
                    page_content=row["text"],
                    metadata={' '.join(c for c in col.split('_')): str(row[col]) for col in df.columns} | {"priority": "high", "tags":tag}
                )
                for _, row in df.iterrows()
            ]

            if documents:
                vectorstore.add_documents(documents)
                print(f"✅ Added {len(documents)} records from '{table}' into ChromaDB!")
        except Exception as e:
            print(f"Unable to populate add {table} table: {str(e)}")
            traceback.print_exc()

    def _process_related_table(self, vectorstore, table, alias=None, parent_table=None, parent_fk=None):
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
        
        tag = self.filter_tag(table)
        documents = [
            Document(
                page_content=row["text"],
                metadata={' '.join(c for c in col.split('_')): str(row[col]) for col in merged_df.columns} | {"priority": "high", "tags":tag}
            )
            for _, row in merged_df.iterrows()
        ]

        if documents:
            vectorstore.add_documents(documents)
            print(f"✅ Added {len(documents)} merged records from '{table}' into ChromaDB!")
            
    def populate_chroma_db(self):
        """Main function to populate ChromaDB with structured database data & FAQ documents."""

        try:
            table_names = self.inspector.get_table_names()
            vectorstore = initial.VECTOR_DB['database'](initial.COLLECTION_NAME)

            for table in self.config.get('relational_tables'):
                if table not in table_names:
                    print(f"⚠️ Table '{table}' does not exist. Skipping...")
                    continue
                self._process_related_table(vectorstore, table)
                
            for table in self.config.get('independent_tables'):
                if table not in table_names:
                    print(f"⚠️ Table '{table}' does not exist. Skipping...")
                    continue
                self._process_independent_table(vectorstore, table)

            # Load FAQ data after database population
            self._load_documents_data()

        except Exception as e:
            traceback.print_exc()
            print(f"❌ Database error: {e}")

        print("🚀 ChromaDB population complete!")
        
    def filter_tag(self, tablename:str):
        order_pattern = initial.CHROMA_FILTER_PATTERNS['order_pattern']
        product_pattern = initial.CHROMA_FILTER_PATTERNS['product_pattern']
        category_pattern = initial.CHROMA_FILTER_PATTERNS['category_pattern']

        tag = 'None'
        if re.search(order_pattern, tablename, re.IGNORECASE):
            tag = 'order_tag'
        elif re.search(product_pattern, tablename, re.IGNORECASE):
            tag = 'product_tag'
        elif re.search(category_pattern, tablename, re.IGNORECASE):
            tag = 'category_tag'
        
        return tag

    
class DummyRetriever(BaseRetriever):
    filtered_docs: list[Document] #add type hinting.

    def _invoke(self, query: str) -> list[Document]:
        return self.filtered_docs

    def _get_relevant_documents(self, query: str) -> list[Document]:
        return self.filtered_docs
    