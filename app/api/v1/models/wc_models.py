from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from app.core.db import Base

class WooCommerceCredential(Base):
    __tablename__ = "woocommerce_credentials"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    customer_id = Column(String)
    customer_secret = Column(String)

class WordPressCredential(Base):
    __tablename__ = "wordpress_credentials"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    username = Column(String)
    password = Column(String)

class WooTableSelection(Base):
    __tablename__ = "woo_table_selections"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    table_name = Column(String)