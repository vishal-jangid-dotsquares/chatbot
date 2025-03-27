import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
import redis
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import RedisChatMessageHistory

# Initialize FastAPI
app = FastAPI()

# Load API Key from Environment Variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Initialize the chat model
model = init_chat_model("llama3-8b-8192", model_provider="groq",max_tokens=100, api_key=GROQ_API_KEY)

# Redis client setup
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

def get_memory(session_id: str):
    """Retrieve conversation memory from Redis"""
    message_history = RedisChatMessageHistory(
        session_id=session_id,
        url="redis://localhost:6379/0"
    )
    return ConversationBufferMemory(chat_memory=message_history, return_messages=True)

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a chatbot assistant for an ecommerce website, that reply the user's queries. Answer all questions to the best of your ability in {language}. Whenever user greet you or ask to tell about yourself then always reply that - i'm a python chat bot and i'm here to solve your query. "),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define input model
class ChatInput(BaseModel):
    message: str
    session_id: str
    language: str = "en"

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

    # Retrieve memory for the session
    memory = get_memory(input.session_id)

    # Append user message to memory
    memory.chat_memory.add_user_message(input.message)

    # Trim messages to fit within model's token limit
    trimmed_messages = trim_messages(
        messages=memory.buffer,
        max_tokens=256,  # Increased limit for better context retention
        strategy="last",  # Keep latest messages
        token_counter=lambda texts: sum(len(text.text().split()) for text in texts),  # Corrected token counter function
        include_system=True,
        allow_partial=False,
        start_on="human",
    )

    # Generate the final prompt
    prompt = prompt_template.invoke({"messages": trimmed_messages, "language": input.language})

    # Generate response from model
    response = model.invoke(prompt)

    # Store AI response in memory
    memory.chat_memory.add_ai_message(response.content)

    return {"response": response.content}

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
