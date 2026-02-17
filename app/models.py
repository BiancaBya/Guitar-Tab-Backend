from sqlalchemy import Column, Integer, String, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)

    tabs = relationship("Tablature", back_populates="owner")

class Tablature(Base):
    __tablename__ = "tablatures"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    json_content = Column(Text) 
    
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="tabs")

    