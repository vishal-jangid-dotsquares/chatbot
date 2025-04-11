from pydantic import BaseModel
from typing import Optional

class FileOut(BaseModel):
    id: int
    filename: str
    filepath: str

    class Config:
        orm_mode = True
