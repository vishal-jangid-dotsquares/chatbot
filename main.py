import asyncio
import uvicorn
import nest_asyncio
from fastapi import FastAPI

from chatbot.rag import Rag
from chroma_handler import ChromaDBPopulator
from chatbot.models import ChatInput

# Initialize FastAPI
app = FastAPI()


@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

    rag_instance = Rag(input)
    response = await rag_instance.invoke()
    return {'response' :response}


async def main():
    populator = ChromaDBPopulator()
    await populator.populate_chroma_db()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
# Run the application
if __name__ == "__main__":
    # It allows to run nested event loops
    # first loop of asyncio and second one of uvicorn
    nest_asyncio.apply()
    asyncio.run(main())

