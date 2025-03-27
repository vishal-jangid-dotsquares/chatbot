from sqlalchemy import inspect
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
from .database import SQLITE_DATABASE_ENGINE
from langchain_core.documents import Document

def populate_chroma_db():
    persist_directory = "./chroma_db"

    # Load embedding model
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Load existing ChromaDB (or create if it doesn't exist)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)

    # Step 1: Load FAQ Data from Text File
    try:
        loader = TextLoader("faq.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        if docs:
            vectorstore.add_documents(docs)
            print(f"📂 Added {len(docs)} FAQ documents from 'faq.txt'")

    except Exception as e:
        print(f"⚠️ FAQ loading error: {e}")

    # Step 2: Connect to Database using SQLAlchemy
    try:
        inspector = inspect(SQLITE_DATABASE_ENGINE)  # SQLAlchemy inspector to get table names

        # List of tables to process (you can modify this list)
        table_names = ["products", "orders", "users"]

        for table in table_names:
            # Check if table exists
            if table not in inspector.get_table_names():
                print(f"⚠️ Table '{table}' does not exist. Skipping...")
                continue

            # Fetch data dynamically
            query = f"SELECT * FROM {table}"
            df = pd.read_sql(query, SQLITE_DATABASE_ENGINE)

            if df.empty:
                print(f"⚠️ Table '{table}' has no data. Skipping...")
                continue

            # Convert each row into a readable format
            df["text"] = df.apply(lambda row: ", ".join(f"{col}: {row[col]}" for col in df.columns), axis=1)

            # Convert to LangChain Document format
            table_docs = [
                Document(
                    page_content=row["text"],
                    metadata={col: str(row[col]) for col in df.columns}  # Store metadata
                )
                for _, row in df.iterrows()
            ]

            # ✅ Store data in ChromaDB
            if table_docs:
                vectorstore.add_documents(table_docs)
                print(f"✅ Added {len(table_docs)} records from '{table}' into ChromaDB!")

    except Exception as e:
        print(f"❌ Database error: {e}")

    print("🚀 ChromaDB population complete!")
    

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=HuggingFaceEmbeddings()  # You can replace with another embedding model
)
retriever = vectorstore.as_retriever()


def retrieve_response(llm, memory):

    # Create a RetrievalQA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever
    )
    # qa_chain = ConversationChain(
    #     llm=llm,
    #     memory=memory,
    #     retriever=retriever
    # )
    # qa_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=retriever,
    #     memory=memory,
    #     return_source_documents=True  # Optional: If you need to return sources
    # )
    return qa_chain
