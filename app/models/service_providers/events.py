from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import date, time

class Event(BaseModel):
    event_name: str
    date: date
    time: time
    venue: str
    event_lat: float
    event_lon: float
    description: str
    post: str
    event_image: Optional[List[str]] = None
    event_video: Optional[List[str]] = None
    email: EmailStr

class EventResponse(Event):
    id: str