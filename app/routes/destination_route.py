from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from app.models.destination import DestinationOut
from app.utils.crud_utils import get_all, get_by_id, delete_by_id
from app.utils.destinationUtils import add_destination_record, update_destination_record
from app.database.connection import destination_collection
from typing import Optional, List
import requests
from app.services.pineconeService import get_pinecone_service
import logging

logger = logging.getLogger(__name__)
OSRM_BASE_URL = "http://router.project-osrm.org"
router = APIRouter()
pinecone_service = get_pinecone_service()

#  Background task - delete data from Pinecone
def delete_from_pinecone_background(destination_id: str):    
    try:
        pinecone_service.delete_destination(destination_id)
        logger.info(f"Successfully deleted destination {destination_id} from Pinecone")
    except Exception as e:
        logger.error(f"Failed to delete from Pinecone: {str(e)}")

#  Background task - sync data to Pinecone ( add & update)
def sync_to_pinecone_background(destination_id: str, destination_data: dict):
    
    try:
        pinecone_service.upsert_destination(destination_id, destination_data)
        logger.info(f"Successfully synced destination {destination_id} to Pinecone")

    except Exception as e:
        logger.error(f"Failed to sync to Pinecone: {str(e)}")

# add destination
@router.post("/", response_model=DestinationOut)
def create_destination(
    background_tasks: BackgroundTasks,
    destination_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    district_name: str = Form(...),
    description: str = Form(...),
    destination_image: List[UploadFile] = File(...),
    category_name: str = Form(...)
):
    destination_data = {
        "destination_name": destination_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name,
        "description": description,
        "category_name": category_name,

        # # Initialize rating fields
        "average_rating": 0.0,
        "total_reviews": 0,
        "rating_breakdown": {
            "5": 0,
            "4": 0,
            "3": 0,
            "2": 0,
            "1": 0
        }
    }
    
    record = add_destination_record(destination_data, images=destination_image)

    background_tasks.add_task(
        sync_to_pinecone_background,
        record['id'],
        record
    )

    return DestinationOut(**record)

# get destinations by id
@router.get("/{destination_id}")
def get_destination_byId(destination_id: str):
    return get_by_id(destination_collection, destination_id)

# update destination by id
@router.put("/{destination_id}", response_model=DestinationOut)
def update_destination(
    destination_id: str,
    background_tasks: BackgroundTasks,
    destination_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    district_name: str = Form(...),
    description: str = Form(...),
    category_name: str = Form(...),
    new_images: Optional[List[UploadFile]] = File(None),
    remove_existing: Optional[List[str]] = Form(None)
):
    doc_ref = destination_collection.document(destination_id)
    updated_record = update_destination_record(
        doc_ref=doc_ref,
        destination_id=destination_id,
        collection=destination_collection,   # 🔹 pass destination collection
        destination_name=destination_name,
        latitude=latitude,
        longitude=longitude,
        district_name=district_name,
        description=description,
        category_name=category_name,
        new_images=new_images,
        remove_existing=remove_existing
    )

    background_tasks.add_task(
        sync_to_pinecone_background,
        destination_id,
        updated_record
    )

    return updated_record
    
# delete destination
@router.delete("/{destination_id}")
def delete_destination(
    destination_id: str,
    background_tasks: BackgroundTasks
):
    files_mapping = {
        "destination_image": "destination_images",
    }

    result = delete_by_id(destination_collection, destination_id, files_mapping)
    background_tasks.add_task(
        delete_from_pinecone_background,
        destination_id
    )

    return result

# get all destinations
@router.get("/")
def get_all_destinations():
    return get_all(destination_collection)

# get destination by district name
@router.get("/district/{district_name}", response_model=list[DestinationOut])
def get_destination_byDistrict(district_name: str):

    destinations = destination_collection.where(
        "district_name_lower", "==", district_name.lower()
    ).stream()

    result = []
    for doc in destinations:
        data = doc.to_dict()
        data["id"] = doc.id
        result.append(data)
    
    if not result:
        raise HTTPException(status_code=404, detail=f"No destinations found in district '{district_name}'")

    return result

# get nearby locations
def get_osrm_distance(lat1, lon1, lat2, lon2):
    """Get road distance in meters between two points using OSRM API"""
    url = f"{OSRM_BASE_URL}/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        if "routes" in data and len(data["routes"]) > 0:
            return data["routes"][0]["distance"], data["routes"][0]["duration"]
        else:
            return None, None
    except Exception as e:
        print(f"OSRM error: {e}")
        return None, None

@router.get("/near/nearby", response_model=list[dict])
def get_nearBy(
    latitude: float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_range: float = 10  # in km
):
    destinations = destination_collection.stream()
    result = []

    for doc in destinations:
        data = doc.to_dict()
        dest_latitude = data.get("latitude")
        dest_longitude = data.get("longitude")

        if dest_latitude is not None and dest_longitude is not None:
            distance, duration = get_osrm_distance(latitude, longitude, dest_latitude, dest_longitude)

            if distance is not None:
                distance_km = distance / 1000  # convert meters to km

                print(f"Destination: {data.get('destination_name')}, Distance: {distance_km:.2f} km")

                if distance_km <= radius_range:
                    data["id"] = doc.id
                    data["distance"] = round(distance_km, 2)
                    data["duration_minutes"] = round(duration / 60, 1) if duration else None
                    result.append(data)

    result.sort(key=lambda x: x['distance'])  # sort by closest first

    if not result:
        raise HTTPException(status_code=404, detail=f"No destinations found within {radius_range} km.")

    return result


# add data to pinecone first time
@router.post("/syncPinecone")
async def bulk_sync_to_pinecone(background_tasks: BackgroundTasks):
    """
    One-time sync: Upload all existing Firebase destinations to Pinecone
    Use this endpoint to initialize Pinecone with existing data
    """
    try:
        destinations = destination_collection.stream()
        count = 0
        
        for doc in destinations:
            data = doc.to_dict()
            data['id'] = doc.id
            
            # Add sync task to background
            background_tasks.add_task(
                sync_to_pinecone_background,
                doc.id,
                data
            )
            count += 1
        
        return {
            "message": f"Queued {count} destinations for Pinecone sync...",
            "status": "processing_in_background"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bulk sync failed: {str(e)}")
