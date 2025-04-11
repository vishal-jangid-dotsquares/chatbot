from pydantic import BaseModel, HttpUrl
from typing import List

class URLInput(BaseModel):
    url: HttpUrl

class URLListOut(BaseModel):
    links: List[HttpUrl]

class ContentInput(BaseModel):
    urls: List[HttpUrl]

class PageContentOut(BaseModel):
    id: int
    url: str
    content: str

    class Config:
        orm_mode = True
