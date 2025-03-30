import os
import redis.asyncio as redis
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.chat_models import init_chat_model

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize the chat model
summarizing_model = init_chat_model(
    "qwen-2.5-coder-32b", 
    model_provider="groq",
    api_key=GROQ_API_KEY
)

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)


class CustomChatMemory:
    
    def __init__(self, max_history=10, expiry=259200):
        
        self.redis_client = redis_client
        self.max_history = max_history
        self.expiry = expiry  # 3 days in seconds

    def add_user(self, user_id:str):
        self.user_id = f"chat_memory:{user_id}"

    async def get_memory(self):
        """Retrieve stored conversation history."""
        history = await self.redis_client.lrange(self.user_id, 0, -1)
        return history or []

    async def add_memory(self, user_message: str, bot_response: str):
        """Store a new conversation pair and summarize if needed."""
        conversation = f"User: {user_message} | Bot: {bot_response}"
        await self.redis_client.rpush(self.user_id, conversation)
        await self.redis_client.ltrim(self.user_id, -self.max_history, -1)  # Maintain max_history size
        await self.redis_client.expire(self.user_id, self.expiry)  # Set expiry every time

        history_length = await self.redis_client.llen(self.user_id)
        if history_length >= self.max_history:
            await self.summarize_memory()
            
    async def summarize_memory(self):
        """Summarize the last three conversations and store as context."""
        history = await self.get_memory()
        if len(history) >= self.max_history:
            summary = await self.summarize_text(history[-self.max_history:])
            await self.redis_client.delete(self.user_id)  # Clear existing history
            await self.redis_client.rpush(self.user_id, summary)  # Store summarized history
            await self.redis_client.ltrim(self.user_id, -self.max_history, -1)  # Ensure list size

    async def summarize_text(self, conversations):
        """Summarize given conversations using LangChain."""
        chain = load_summarize_chain(summarizing_model, chain_type="map_reduce")
        docs = [Document(page_content=conv) for conv in conversations]
        response = await chain.ainvoke({"input_documents": docs})
        summary = response["output_text"]
        return " ".join(summary.split()[:200])  # Ensure max 50 words, preserving sentence structure

    async def get_conversation(self):
        """Attach stored conversation context to a new prompt in a structured way."""
        history = await self.get_memory()
        structured_context = "\n".join([f"{conv}" for conv in history]) if history else ""
        return structured_context
