from pydantic import BaseModel
from sqlalchemy import Column, String, Integer
from database.database import Base

class UserDB(Base):
    __tablename__ = "users"
    userId = Column(Integer, primary_key=True, index=True)
    userName = Column(String, nullable=False)
    userEmail = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

class CreateUserRequest(BaseModel):
    userName: str
    userEmail: str
    password: str

class UserResponse(BaseModel):
    userId: int
    userName: str
    userEmail: str
