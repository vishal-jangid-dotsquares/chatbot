from fastapi import FastAPI
from core.rag import Rag
from core.models import ChatInput

# Initialize FastAPI
app = FastAPI()
from fastapi.responses import StreamingResponse
@app.post("/chat/")
async def chat(input: ChatInput):
    """Handles user queries and maintains conversation history."""

    rag_instance = Rag(input)
    response = await rag_instance.invoke()
    async def streamable_response():
        async for chunk in response.astream({}):
            yield chunk
    return StreamingResponse(streamable_response(), media_type="text/plain")
