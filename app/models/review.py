from pydantic import BaseModel, Field, EmailStr
from typing import Literal, Optional

class ReviewCreate(BaseModel):
    user_email: EmailStr = Field(..., description="User email for authentication")
    reviewable_type: Literal["destination", "service"]
    reviewable_id: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    title: str = Field(..., min_length=3, max_length=255)
    comment: str = Field(..., min_length=10)
    visit_date: Optional[str] = None  # ISO format date

class ReviewUpdate(BaseModel):
    rating: Optional[int] = Field(None, ge=1, le=5)
    title: Optional[str] = Field(None, min_length=3, max_length=255)
    comment: Optional[str] = Field(None, min_length=10)
    visit_date: Optional[str] = None

class ReviewResponse(BaseModel):
    id: str
    user_id: str
    user_name: str
    user_email: str
    reviewable_type: str
    reviewable_id: str
    rating: int
    title: str
    comment: str
    visit_date: Optional[str] = None
    helpful_count: int = 0
    is_verified: bool = False
    status: str = "approved"
    created_at: str
    updated_at: str

class HelpfulRequest(BaseModel):
    user_email: EmailStr

# use for content generation - feedback system
class FeedbackRequest(BaseModel):
    feedback: str = Field(..., pattern="^(excellent|good|acceptable|poor|incorrect)$")