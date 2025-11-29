import httpx, asyncio
from app.config.settings import settings
from typing import List

INTEREST_TO_DESTINATION_CATEGORY = {
    "adventure sports": ["Adventure", "Mountains", "Beach"],
    "cultural sites": ["Religious", "Historical"],
    "nature and wildlife": ["Wildlife", "Natural Park", "Mountains"],
    "museums and art": ["Historical"],
    "beaches and relaxation": ["Beach"],
    "nightlife": ["Beach", "Historical"], 
    "photography": ["Natural Park", "Mountains", "Beach", "Historical"],
    "local experience": ["Religious", "Historical"],
    "historical places": ["Historical", "Religious"],
}

def get_destination_categories_for_interests(interests: List[str]) -> List[str]:
    """
    Map user interests to destination categories in database
    Returns unique list of categories to filter in Pinecone
    """
    categories = set()
    for interest in interests:
        mapped = INTEREST_TO_DESTINATION_CATEGORY.get(interest.lower(), [])
        categories.update(mapped)
    
    return list(categories) if categories else ["Religious", "Historical", "Beach", "Adventure", "Mountains", "Wildlife", "Natural Park"]

async def get_osrm_route(lat1: float, lon1: float, lat2: float, lon2: float, max_retries: int = 3) -> dict:
   
    osrm_url = settings.OSRM_URL   
    url = f"{osrm_url}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}"
    params = {
        "overview": "false",
        "steps": "false"
    }
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("code") == "Ok" and data.get("routes"):
                        route = data["routes"][0]
                        distance_m = route["distance"]
                        duration_s = route["duration"]
                        
                        return {
                            "distance_km": round(distance_m / 1000, 2),
                            "duration_mins": int(duration_s / 60)
                        }
                
                # Retry on non-200 status
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5 * (attempt + 1))  
                    continue
                    
                raise Exception(f"OSRM request failed: {response.status_code}")
                
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise Exception(f"OSRM timeout after {max_retries} attempts")
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue
            raise Exception(f"Error calling OSRM: {e}")


def get_visit_duration(category: str, group_type: str) -> int:
    """
    Get typical visit duration in minutes based on category and group type
    """
    base_durations = {
        "religious": 90,
        "historical": 120,
        "beach": 180,
        "adventure": 180,
        "mountains": 150,
        "wildlife": 180,
        "natural park": 150,
        # Service durations
        # "restaurant": 90,
        # "cafe": 45,
        # "spa": 120,
        # "shopping": 90,
        "nightlife": 120
    }
    
    duration = base_durations.get(category.lower(), 120)
    
    # Adjust for group type
    if group_type == "family":
        duration = int(duration * 1.2)
    elif group_type == "solo":
        duration = int(duration * 0.8)
    elif group_type == "couple":
        duration = int(duration * 0.9)
    
    return duration


def get_transport_speed(transport_mode: str) -> float:
    """
    Get average speed in km/h for different transport modes
    Used for fallback calculations
    """
    speeds = {
        "bike": 15,
        "car": 50,
        "van": 45,
        "bus": 40,
        "train": 60
    }
    return speeds.get(transport_mode.lower(), 40)