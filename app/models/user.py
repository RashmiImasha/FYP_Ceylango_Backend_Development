from pydantic import BaseModel, EmailStr, Field
from typing import Dict, List, Literal, Optional



# Base models
class UserBase(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "tourist", "service_provider"] = "tourist"

class UserCreate(UserBase):
    password: str

class ServiceProviderApplication(BaseModel):
    email: EmailStr
    full_name: str
    service_name: str
    district: str
    service_category: str
    phone_number: str

    description: Optional[str] = None
    
    class Config:
        use_enum_values = True


class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserInDB(UserBase):
    uid: str
    disabled: Optional[bool] = False
    

# Service Provider Profile Models
class BaseServiceProfile(BaseModel):
    service_name: str
    service_category: str
    description: str
    address: str
    district: str
    coordinates: Optional[Dict[str, float]] = None  # {"lat": 0.0, "lng": 0.0}
    phone_number: str
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    social_media: Optional[Dict[str, str]] = None
    operating_hours: Dict[str, Dict[str, str]]  # {"monday": {"open": "09:00", "close": "17:00"}}
    
    images: List[str] = []  # URLs to images
    poster_images: List[str] = []   # Promotional poster images

    amenities: List[str] = []
    is_active: bool = True
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    
    class Config:
        use_enum_values = True

# Update Profile Request Models
class UpdateProfileBasicInfo(BaseModel):
    service_name: str
    description: str
    address: str
    district: str
    phone_number: str
    email: Optional[EmailStr] = None
    website: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None


class TimeSlot(BaseModel):
    open: str
    close: str

class UpdateOperatingHours(BaseModel):
    operating_hours: Dict[str, TimeSlot]


class UpdateSocialMedia(BaseModel):
    social_media: Dict[str, str]


class UpdateAmenities(BaseModel):
    amenities: List[str]