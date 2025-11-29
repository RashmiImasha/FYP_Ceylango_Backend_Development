from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime

class ChatSession(BaseModel):
    session_id: str
    user_id: Optional[str] = None
    messages: List[ChatMessage] = []
    created_at: datetime
    updated_at: datetime

class ChatRequest(BaseModel):
    query: str
    lat: float | None = None
    lon: float | None = None
    session_id: str | None = None  # for chat history
    user_id: str | None = None 