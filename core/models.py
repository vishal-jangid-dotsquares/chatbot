from pydantic import BaseModel


# Define input model
class ChatInput(BaseModel):
    message: str
    session_id: str