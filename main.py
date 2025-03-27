from fastapi import FastAPI
from chatbot.rag import Rag
from chatbot.models import ChatInput

# Initialize FastAPI
app = FastAPI()

@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

    rag_instance = Rag(input)
    response = await rag_instance.invoke()
    return {'response' : response}
