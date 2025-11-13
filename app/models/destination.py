
from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, List

class Destination(BaseModel):
    destination_name: str
    latitude: float
    longitude: float
    district_name: str
    description: str 
    destination_image: List[str]
    category_name: str
    image_phash: List[str] = None
    district_name_lower: Optional[str] = None
    average_rating: float = 0.0
    total_reviews: int = 0
    rating_breakdown: Dict[str, int] = {
        "1": 0,
        "2": 0,
        "3": 0, 
        "4": 0,
        "5": 0
    }
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
class DestinationOut(Destination):
   id: str
   
   model_config = ConfigDict(
       exclude = {'image_phash', 'district_name_lower'}
   )

class DestinationNearBy(DestinationOut):
    distance: float # in km
    

class MissingPlaceOut(Destination):
    id: str

    model_config = ConfigDict(
        exclude={'image_phash', 'district_name_lower'}
    )