from pydantic import BaseModel, ConfigDict
from typing import Optional, List

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
    
class DestinationOut(Destination):
   id: str
   
   model_config = ConfigDict(
       exclude = {'image_phash', 'district_name_lower'}
   )

class DestinationNearBy(DestinationOut):
    distance: float # in km