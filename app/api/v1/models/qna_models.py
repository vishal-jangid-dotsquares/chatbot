from sqlalchemy import Column, Integer, String, ForeignKey
from app.db import Base

class QnA(Base):
    __tablename__ = "qna"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    ques = Column(String)
    answer = Column(String)
