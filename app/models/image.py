from pydantic import BaseModel
from typing import List

class SnapImageResponse(BaseModel):
    destination_name: str
    district_name: str
    historical_background: str
    cultural_significance: str
    what_makes_it_special: str
    visitor_experience: str
    interesting_facts: List[str]
    request_id: str

    