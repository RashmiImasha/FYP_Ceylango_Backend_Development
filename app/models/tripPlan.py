from pydantic import BaseModel, validator
from typing import Any, Dict, Optional, List
from enum import Enum
from datetime import datetime


class InterestType(str, Enum):
    ADVENTURE_SPORTS = "adventure sports"
    CULTURAL_SITES = "cultural sites"
    NATURE_WILDLIFE = "nature and wildlife"
    MUSEUMS_ART = "museums and art"
    BEACHES_RELAXATION = "beaches and relaxation"
    NIGHTLIFE = "nightlife"
    PHOTOGRAPHY = "photography"
    LOCAL_EXPERIENCE = "local experience"
    HISTORICAL_PLACES = "historical places"

class TransportMode(str, Enum):
    BIKE = "bike"
    CAR = "car"
    VAN = "van"
    BUS = "bus"
    TRAIN = "train"

class GroupType(str, Enum):
    SOLO = "solo"
    COUPLE = "couple"
    FAMILY = "family"
    FRIENDS = "friends"

class ActivitySlot(BaseModel):
    start_time: str
    end_time: str
    destination_id: Optional[str] = None
    destination_name: Optional[str] = None    
    activity_type: str  # visit, travel, meal, service
    duration_mins: int
    travel_from_previous_mins: int
    tips: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None 
    category: Optional[str] = None
    rating: Optional[float] = None 

class DayItinerary(BaseModel):
    day_number: int
    date: str
    theme: str
    activities: List[ActivitySlot]
    total_distance_km: float
    total_travel_time_mins: int

class TripPlanRequest(BaseModel):
    user_id: str
    districts: List[str]
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    group_type: GroupType
    interests: List[InterestType]
    transport_mode: TransportMode
    
    @validator('end_date')
    def validate_dates(cls, v, values):
        if 'start_date' in values:
            start = datetime.strptime(values['start_date'], '%Y-%m-%d')
            end = datetime.strptime(v, '%Y-%m-%d')
            if end <= start:
                raise ValueError('end_date must be after start_date')
        return v
    
    @validator('interests')
    def validate_interests(cls, v):
        if not v or len(v) == 0:
            raise ValueError('At least one interest must be provided')
        if len(v) > 5:
            raise ValueError('Maximum 5 interests allowed')
        return v

class TripPlanResponse(BaseModel):
    trip_id: str
    trip_name: str    
    summary: Dict[str, Any]
    itinerary: List[DayItinerary]
    map_data: Dict[str, Any]
    alternatives: Optional[List[Dict]] = None
    generated_at: str
