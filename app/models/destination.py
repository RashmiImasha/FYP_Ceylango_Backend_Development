from pydantic import BaseModel
from typing import Optional

class Destination(BaseModel):
    destination_name: str
    latitude: float
    longitude: float
    district_name: str
    description: str 
    destination_image: str
    category_id: str
    image_hash: Optional[str] = None
    
class DestinationOut(Destination):
   id: str
   category_name: str