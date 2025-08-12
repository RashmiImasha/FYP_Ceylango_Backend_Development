from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query
from app.models.destination import DestinationOut, DestinationNearBy
from app.utils.destinationUtils import haversine, compute_phash
from firebase_admin import storage
from app.database.connection import db
from typing import Optional
import uuid, imagehash
from urllib.parse import urlparse

# create router for destination routes
router = APIRouter()
destination_collection = db.collection('destination')
category_collection = db.collection('category')

# add destination
@router.post("/", response_model=DestinationOut)
def create_destination(
    destination_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    district_name: str = Form(...),
    description: str = Form(...),
    destination_image: UploadFile = File(...),
    category_name: str = Form(...)
):
    destination_name = destination_name.strip()
    district_name = district_name.strip()
    category_name = category_name.strip()
    
    allow_imageType = ['image/jpeg', 'image/png']
    if destination_image.content_type not in allow_imageType:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Check if category exists
    category_query = category_collection \
        .where('category_name', '==', category_name) \
        .where('category_type', '==', 'location') \
        .get()
    
    if not category_query:
        raise HTTPException(status_code=404, detail="Category not found or not of type 'location'")    
   
    # check if destination name already exists
    existing_destination = destination_collection.where('destination_name', '==', destination_name).get()
    if existing_destination:
        raise HTTPException(status_code=400, detail="This Destination name already exists...!")

    # check nearby locations_within ~5 meters
    existing_destinations = destination_collection.stream()
    for doc in existing_destinations:
        data = doc.to_dict()
        existing_lat = data.get("latitude")
        existing_lon = data.get("longitude")

        if existing_lat is not None and existing_lon is not None:
            distance = haversine(latitude, longitude, existing_lat, existing_lon)
            if distance < 5:  # can change this to 10 or 20 if you prefer a wider buffer
                raise HTTPException(status_code=400, detail="A destination already exists very close to this location")
            
    # compute pHash of the image
    destination_image.file.seek(0)  
    image_phash = compute_phash(destination_image.file)

    # Check for visually similar image (threshold <= 5 is a good start)
    existing_dest_images = destination_collection.stream()
    for doc in existing_dest_images:
        existing_phash = doc.to_dict().get('image_phash')
        if existing_phash:
            distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(image_phash)
            if distance <= 5:  # Adjust threshold as needed
                raise HTTPException(status_code=400, detail="A visually similar image already exists.")
    
    # upload image to firebase storage
    destination_image.file.seek(0)  # Reset file pointer
    bucket = storage.bucket()
    image_id = str(uuid.uuid4())
    blob = bucket.blob(f'destination_images/{image_id}_{destination_image.filename}')
    blob.upload_from_file(destination_image.file, content_type=destination_image.content_type)
    blob.make_public()

    image_url = blob.public_url

    # Add new destination
    destination_data = {
        "destination_name": destination_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name,
        "district_name_lower": district_name.lower(),
        "description": description,
        "destination_image": image_url,
        "image_phash": image_phash,
        "category_name": category_name
    }

    _, doc_ref = destination_collection.add(destination_data)
    return DestinationOut(id=doc_ref.id, **destination_data)

# get destinations by id
@router.get("/{destination_id}", response_model=DestinationOut)
def get_destination_byId(destination_id: str):
    doc_ref = destination_collection.document(destination_id)
    destination = doc_ref.get()
    
    if not destination.exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    destination_data = destination.to_dict()
    destination_data['id'] = destination.id 
    return destination_data

# update destination by id
@router.put("/{destination_id}", response_model=DestinationOut)
def update_destination(
    destination_id: str,
    destination_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    district_name: str = Form(...),
    description: str = Form(...),
    destination_image: Optional[UploadFile] = File(None),
    category_name: str = Form(...)
):
    destination_name = destination_name.strip()
    district_name = district_name.strip()
    category_name = category_name.strip()

    doc_ref = destination_collection.document(destination_id)
    destination = doc_ref.get()
    
    if not destination.exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    current_data = destination.to_dict()
    
    # Check if category exists
    matching_category = category_collection.where('category_name', '==', category_name).stream()
    category_doc = None
    for doc in matching_category:
        cat_data = doc.to_dict()
        if cat_data.get('category_type') == 'location':
            category_doc = doc
            break
    
    if not category_doc:
        raise HTTPException(status_code=404, detail="Valid location category not found")    
    
    category_name = category_doc.to_dict().get("category_name")

    # Check if destination name already exists
    existing_destination = destination_collection.where('destination_name', '==', destination_name).stream()
    for doc in existing_destination:
        if doc.id != destination_id:
            raise HTTPException(status_code=400, detail="This Destination name already exists...!")

    
    # Reject if updated location is too close to another existing destination
    existing_destinations = destination_collection.stream()
    for doc in existing_destinations:
        if doc.id == destination_id:
            continue
        data = doc.to_dict()
        existing_lat = data.get("latitude")
        existing_lon = data.get("longitude")
        if existing_lat is not None and existing_lon is not None:
            distance = haversine(latitude, longitude, existing_lat, existing_lon)
            if distance < 5:  
                raise HTTPException(status_code=400, detail="Another destination already exists very close to this location")

    # check image type
    if destination_image:
        allow_imageType = ['image/jpeg', 'image/png']
        if destination_image.content_type not in allow_imageType:
            raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # upload image to firebase storage
    image_url = current_data.get("destination_image")
    image_phash = current_data.get("image_phash")

    if destination_image:
        destination_image.file.seek(0)
        new_phash = compute_phash(destination_image.file)

        # Check if visually similar image already exists
        existing_dest_images = destination_collection.stream()
        for doc in existing_dest_images:
            if doc.id == destination_id:
                continue  # Skip self
            existing_phash = doc.to_dict().get('image_phash')
            if existing_phash:
                distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(new_phash)
                if distance <= 5:
                    raise HTTPException(status_code=400, detail="A visually similar image already exists.")

        image_phash = new_phash

        # Delete old image
        if image_url:
            parsed_url = urlparse(image_url)
            image_path = parsed_url.path.lstrip('/')
            blob = storage.bucket().blob(image_path)
            if blob.exists():
                blob.delete()

        # Upload new image
        destination_image.file.seek(0)
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'destination_images/{image_id}_{destination_image.filename}')
        blob.upload_from_file(destination_image.file, content_type=destination_image.content_type)
        blob.make_public()
        image_url = blob.public_url

        
    # Update destination data
    destination_data = {
        "destination_name": destination_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name,
        "district_name_lower": district_name.lower(),
        "description": description,
        "destination_image": image_url,
        "image_phash": image_phash,
        "category_name": category_name
    }

    doc_ref.update(destination_data)
    
    return {
        "id": destination_id,
        **destination_data,
    }

# delete destination
@router.delete("/{destination_id}", response_model=dict)
def delete_destination(destination_id: str):
    doc_ref = destination_collection.document(destination_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    data = doc.to_dict()

    # delete image from firebase storage 
    image_url = data.get('destination_image')
    image_delete = False
    if image_url:
        parsed_url = urlparse(image_url)
        image_blob_name = parsed_url.path.lstrip('/')
        bucket = storage.bucket()
        blob = bucket.blob(image_blob_name)
        
        if blob.exists():
            blob.delete()
            image_delete = True
    
    doc_ref.delete()
    return {
        "message": "Destination deleted successfully",
        "destination_id": destination_id,
        "image_status": "Image deleted from Firebase Storage" if image_delete else "No image found or already deleted"
    }

# get all destinations
@router.get("/", response_model=list[DestinationOut])
def get_all_destinations():
    destinations = destination_collection.stream()
    result = []

    for doc in destinations:
        data = doc.to_dict()
        data['id'] = doc.id       

        result.append(data)

    return result

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

# get nearby destination ( default : 10km )
@router.get("/near/nearby", response_model=list[DestinationNearBy])
def get_nearBy(
    latitude: float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_range: float = 10 # in km
):
    # print("get_nearBy route called")

    destinations = destination_collection.stream()
    result = []

    # print(f"User coords: {latitude}, {longitude}")

    for doc in destinations:
        data = doc.to_dict()
        dest_latitude = data.get("latitude")
        dest_longitude = data.get("longitude")

        # # check values and types
        # print(f"Destination: {data.get('destination_name')}")
        # print(f"Latitude: {dest_latitude} (type: {type(dest_latitude)})")
        # print(f"Longitude: {dest_longitude} (type: {type(dest_longitude)})")

        if dest_latitude is not None and dest_longitude is not None:
            distance = haversine(latitude, longitude, dest_latitude, dest_longitude)

            print(f"Checking destination: {data.get('destination_name')} "
                  f"at {dest_latitude}, {dest_longitude}")
            print(f"Distance: {distance/1000:.2f} km")

            if distance <= radius_range*1000:
                data["id"] = doc.id
                data["distance"] = round(distance/1000, 2)
                result.append(data)
    
    result.sort(key=lambda x: x['distance'])  # sort by closest first

    if not result:
        raise HTTPException(status_code=404, detail=f"No destinations found within {radius_range} km.")
    
    return result
