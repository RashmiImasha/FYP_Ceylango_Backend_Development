from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from app.models.destination import DestinationOut, DestinationNearBy
from app.utils.crud_utils import get_all, get_by_id, delete_by_id
from app.utils.destinationUtils import haversine, compute_phash, add_destination_record, update_destination_record
from firebase_admin import storage
from app.database.connection import destination_collection
from typing import Optional, List
import imagehash
from urllib.parse import urlparse
import requests

OSRM_BASE_URL = "http://router.project-osrm.org"

# create router for destination routes
router = APIRouter()

# add destination
@router.post("/", response_model=DestinationOut)
def create_destination(
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
        "category_name": category_name
    }
    
    record = add_destination_record(destination_data, images=destination_image)
    return DestinationOut(**record)


# get destinations by id
@router.get("/{destination_id}")
def get_destination_byId(destination_id: str):
    return get_by_id(destination_collection, destination_id)

# update destination by id
@router.put("/{destination_id}", response_model=DestinationOut)
def update_destination(
    destination_id: str,
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
    return update_destination_record(
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

# def update_destination(
#     destination_id: str,
#     destination_name: str = Form(...),
#     latitude: float = Form(...),
#     longitude: float = Form(...),
#     district_name: str = Form(...),
#     description: str = Form(...),
#     category_name: str = Form(...),
#     new_images: Optional[List[UploadFile]] = File(None),
#     remove_existing: Optional[List[str]] = Form(None)  # send URLs of images to remove
# ):
#     doc_ref = destination_collection.document(destination_id)
#     destination = doc_ref.get()
#     if not destination.exists:
#         raise HTTPException(status_code=404, detail="Destination not found")

#     current_data = destination.to_dict()
#     current_data["id"] = destination_id

#     # Validate category
#     category_query = category_collection.where('category_name', '==', category_name).where('category_type', '==', 'location').get()
#     if not category_query:
#         raise HTTPException(status_code=404, detail="Valid location category not found")

#     # Check duplicate destination name
#     existing_destination = destination_collection.where('destination_name', '==', destination_name).stream()
#     for doc in existing_destination:
#         if doc.id != destination_id:
#             raise HTTPException(status_code=400, detail="This Destination name already exists...!")

#     # Check nearby destinations
#     for doc in destination_collection.stream():
#         if doc.id == destination_id:
#             continue
#         data = doc.to_dict()
#         if data.get("latitude") is not None and data.get("longitude") is not None:
#             distance = haversine(latitude, longitude, data["latitude"], data["longitude"])
#             if distance < 5:
#                 raise HTTPException(status_code=400, detail="Another destination is too close to this location")

    
#     current_images = current_data.get("destination_image", []) or []
#     current_phash = current_data.get("image_phash", []) or []

#     # Remove selected images
#     if remove_existing:
#         for url in remove_existing:
#             if url in current_images:
#                 idx = current_images.index(url)
#                 current_images.pop(idx)
#                 current_phash.pop(idx)
#                 delete_file_from_storage({"destination_image": [url]}, {"destination_image": "destination_images"})

#     # Add new images
#     if new_images:
#         allowed_types = ["image/jpeg", "image/jpg", "image/png"]
#         existing_phash_lists = [
#             doc.to_dict().get("image_phash", []) for doc in destination_collection.stream() if doc.id != destination_id
#         ]

#         current_phash_set = set(current_phash)

#         for img in new_images:
#             if img.content_type not in allowed_types:
#                 raise HTTPException(status_code=400, detail="Unsupported image file type")
#             img.file.seek(0)
#             phash = compute_phash(img.file)

#             if phash in current_phash_set:
#                 raise HTTPException(status_code=400, detail="This image already exists in this Image List...!")
            
#             # Check similarity
#             for phash_list in existing_phash_lists:
#                 for existing_phash in phash_list:
#                     if imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(phash) <= 5:
#                         raise HTTPException(status_code=400, detail="A visually similar image already exists.")
#             url = upload_file_to_storage(img, "destination_images")
#             if url:
#                 current_images.append(url)
#                 current_phash.append(phash)
#                 current_phash_set.add(phash)

#     if len(current_images) == 0:
#         raise HTTPException(status_code=400, detail="At least one image must remain")

#     # Update other fields
#     current_data.update({
#         "destination_name": destination_name.strip(),
#         "latitude": latitude,
#         "longitude": longitude,
#         "district_name": district_name.strip(),
#         "district_name_lower": district_name.strip().lower(),
#         "description": description,
#         "category_name": category_name.strip(),
#         "destination_image": current_images,
#         "image_phash": current_phash
#     })

#     doc_ref.set(current_data, merge=False)
#     return {"id": destination_id, **current_data}
    
# delete destination
@router.delete("/{destination_id}")
def delete_destination(destination_id: str):
    files_mapping = {
        "destination_image": "destination_images",
    }
    return delete_by_id(destination_collection, destination_id, files_mapping)

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





# # get nearby destination ( default : 10km )
# @router.get("/near/nearby", response_model=list[DestinationNearBy])
# def get_nearBy(
#     latitude: float = Query(..., description="User's current latitude"),
#     longitude: float = Query(..., description="User's current longitude"),
#     radius_range: float = 10 # in km
# ):
#     # print("get_nearBy route called")

#     destinations = destination_collection.stream()
#     result = []

#     # print(f"User coords: {latitude}, {longitude}")

#     for doc in destinations:
#         data = doc.to_dict()
#         dest_latitude = data.get("latitude")
#         dest_longitude = data.get("longitude")

#         # # check values and types
#         # print(f"Destination: {data.get('destination_name')}")
#         # print(f"Latitude: {dest_latitude} (type: {type(dest_latitude)})")
#         # print(f"Longitude: {dest_longitude} (type: {type(dest_longitude)})")

#         if dest_latitude is not None and dest_longitude is not None:
#             distance = haversine(latitude, longitude, dest_latitude, dest_longitude)

#             print(f"Checking destination: {data.get('destination_name')} "
#                   f"at {dest_latitude}, {dest_longitude}")
#             print(f"Distance: {distance/1000:.2f} km")

#             if distance <= radius_range*1000:
#                 data["id"] = doc.id
#                 data["distance"] = round(distance/1000, 2)
#                 result.append(data)
    
#     result.sort(key=lambda x: x['distance'])  # sort by closest first

#     if not result:
#         raise HTTPException(status_code=404, detail=f"No destinations found within {radius_range} km.")
    
#     return result
