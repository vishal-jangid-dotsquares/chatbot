from typing import List
from pydantic import BaseModel

class WooCredentialCreate(BaseModel):
    customer_id: str
    customer_secret: str

class WordPressCredentialCreate(BaseModel):
    username: str
    password: str

class WooTableInput(BaseModel):
    tables: List[str]

class WooTableOut(BaseModel):
    id: int
    table_name: str

    class Config:
        orm_mode = True
