
from sqlalchemy import Column, Integer, String
from app.database import Base

class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    roll_number = Column(String)
    room_number = Column(String)
    hall = Column(String)
    description = Column(String)
    category = Column(String)
    urgency = Column(String)
    location = Column(String)
    summary = Column(String)
    status = Column(String, default="Pending")  # ✅ Added field

class Manager(Base):
    __tablename__ = "managers"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    hall = Column(String)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
