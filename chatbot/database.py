import os
from typing import Literal
from dotenv import load_dotenv
import psycopg2  # PostgreSQL
import mysql.connector  # MySQL

load_dotenv()

# Database connector 
class DatabaseConnector:
    
    def __init__(self, type:Literal['mysql','postgresql']):
        self.config = {
            "host": os.getenv('DATABASE_HOST'),
            "user": os.getenv('DATABASE_USER'),
            "password": os.getenv('DATABASE_PASSWORD'),
            "name": os.getenv('DATABASE_NAME'),
            "port": os.getenv('DATABASE_PORT'),
            "type":type
        }
        self.connection = None

    def connect(self):
        """Connect to the database dynamically based on the config."""
        self.db_type = self.config["type"]
        print("R...............", self.config)
        try:
            if self.db_type == "postgresql":
                self.connection = psycopg2.connect(
                    host=self.config["host"],
                    port=self.config["port"],
                    user=self.config["user"],
                    password=self.config["password"],
                    dbname=self.config["name"]
                )
                print("✅ Connected to PostgreSQL")
                return self.connection

            elif self.db_type == "mysql":
                self.connection = mysql.connector.connect(
                    host=self.config["host"],
                    port=self.config["port"],
                    user=self.config["user"],
                    password=self.config["password"],
                    database=self.config["name"]
                )
                print("X.............", self.connection)
                print("✅ Connected to MySQL")
                return self.connection
            else:
                raise Exception(f"❌ Unsupported database type: {self.db_type}")

        except Exception as e:
            print(f"❌ Database connection failed: {e}")

        return self.connection

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            print("🔌 Connection closed.")

