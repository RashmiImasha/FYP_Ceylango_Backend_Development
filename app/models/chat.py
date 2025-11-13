# app/models/chat.py

from pydantic import BaseModel
from typing import Optional, Dict, List

class CreateConversationRequest(BaseModel):
    tourist_id: str
    provider_id: str

class ConversationResponse(BaseModel):
    conversation_id: str
    exists: bool
    tourist_name: Optional[str] = None
    provider_name: Optional[str] = None

class ConversationListItem(BaseModel):
    conversation_id: str
    other_user_id: str
    other_user_name: str
    other_user_avatar: str
    other_user_role: str
    last_message: str
    last_message_time: Optional[str]
    unread_count: int
    is_last_message_mine: bool

class UserStatusUpdate(BaseModel):
    online: bool
    fcm_token: Optional[str] = None