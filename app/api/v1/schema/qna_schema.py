from pydantic import BaseModel
from typing import List, Optional

class QnAItem(BaseModel):
    ques: str
    answer: str

class QnAInput(BaseModel):
    qna_list: List[QnAItem]

class QnAOut(BaseModel):
    id: int
    ques: str
    answer: str

    class Config:
        orm_mode = True
