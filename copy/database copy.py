import json
import sqlite3
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentType
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from sqlalchemy import create_engine
import psycopg2  # PostgreSQL
import mysql.connector  # MySQL

DATABASE_URI = "sqlite:///test.db"  # SQLite file-based database
SQLITE_DATABASE_ENGINE = create_engine(DATABASE_URI)

# Connect LangChain to SQLite
db = SQLDatabase(SQLITE_DATABASE_ENGINE, sample_rows_in_table_info=0)


# Initialize the LLM-powered SQL Agent
def get_sql_agent(llm):
    agent = create_sql_agent(
        llm=llm,
        db=db,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        allow_tools=["sql_db_query"],
        return_direct=True
    )
    return agent


# Database connector 
class DatabaseConnector:
    
    def __init__(self, config_path="config.json"):
        self.config = self.load_config(config_path)
        self.connection = None

    def load_config(self, config_path):
        """Load JSON config file safely."""
        try:
            with open(config_path, "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise Exception("Error: Config file not found!")
        except json.JSONDecodeError:
            raise Exception("Error: Invalid JSON format in config file!")

    def connect(self):
        """Connect to the database dynamically based on the config."""
        db_config = self.config["database"]
        db_type = db_config["type"].lower()

        try:
            if db_type == "postgresql":
                self.connection = psycopg2.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    user=db_config["user"],
                    password=db_config["password"],
                    dbname=db_config["name"]
                )
                print("✅ Connected to PostgreSQL")

            elif db_type == "mysql":
                self.connection = mysql.connector.connect(
                    host=db_config["host"],
                    port=db_config["port"],
                    user=db_config["user"],
                    password=db_config["password"],
                    database=db_config["name"]
                )
                print("✅ Connected to MySQL")

            elif db_type == "sqlite":
                self.connection = sqlite3.connect(db_config["sqlite_path"])
                print("✅ Connected to SQLite")

            else:
                raise Exception(f"❌ Unsupported database type: {db_type}")

        except Exception as e:
            print(f"❌ Database connection failed: {e}")

        return self.connection

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Connection closed.")

