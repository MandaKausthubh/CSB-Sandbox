from sqlalchemy import Column, Integer, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from database.database import Base

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    title = Column(String(255), nullable=True)
    input_text = Column(Text, nullable=False)

    parameters = Column(Text)        # JSON string
    result = Column(Text)            # JSON string

    score = Column(Float, nullable=True)
    status = Column(String(30), default="processing")

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class AnalysisCreate(BaseModel):
    content: str
    contentType: str = "text"
    targetAudience: List[str]
    platform: List[str]
    region: List[str]
    sponsors: List[str] = []
    project_id: Optional[int] = None  # still supported

class AnalysisOut(BaseModel):
    id: int
    userId: int
    content: str
    contentType: str
    targetAudience: List[str]
    platform: List[str]
    region: List[str]
    sponsors: List[str]
    score: float
    complianceStatus: str
    violations: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    aiEnhancedScript: str
    createdAt: datetime

    class Config:
        orm_mode = True
