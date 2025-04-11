from pydantic import BaseModel

class ChatInput(BaseModel):
    message: str
    session_id: str
