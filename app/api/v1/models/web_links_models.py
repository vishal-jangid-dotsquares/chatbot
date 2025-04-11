from sqlalchemy import Column, Integer, String, Text, ForeignKey
from app.db import Base

class PageContent(Base):
    __tablename__ = "page_contents"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    url = Column(String)
    content = Column(Text)
