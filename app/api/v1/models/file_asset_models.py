from sqlalchemy import Column, Integer, String, ForeignKey
from app.db import Base

class FileAsset(Base):
    __tablename__ = "file_assets"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    filename = Column(String)
    filepath = Column(String)
