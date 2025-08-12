from pydantic import BaseModel
from typing import Optional

class EmergancyContact(BaseModel):
    id: Optional[str] = None
    police_unit: str
    police_district: str
    police_latitude: float
    police_longitude: float
    police_contact: list[str]
    
class EmergancyContactResponse(BaseModel):
    message: str
    data: EmergancyContact

class EmergancyNearest(EmergancyContact):
    distance_km: float