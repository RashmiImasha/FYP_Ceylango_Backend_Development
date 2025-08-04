from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
from app.database.connection import db
from typing import Optional
import uuid
import math
from PIL import Image
import imagehash

from app.models.destination import Destination, DestinationOut
from urllib.parse import urlparse

# create router for destination routes
router = APIRouter()
destination_collection = db.collection('destination')
category_collection = db.collection('category')

def haversine(lat1, lon1, lat2, lon2):
    # calculate distance between two points on the Earth ( Radius )
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def compute_phash(upload_file) -> str:
    image = Image.open(upload_file)
    phash = imagehash.phash(image)
    return str(phash)


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
    
    category_doc = category_query[0]
    category_id = category_doc.id
    category_data = category_doc.to_dict()

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
            if distance < 5:  # You can change this to 10 or 20 if you prefer a wider buffer
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
        "description": description,
        "destination_image": image_url,
        "image_phash": image_phash,
        "category_id": category_id
    }

    _, doc_ref = destination_collection.add(destination_data)

    return {
        "id": doc_ref.id,
        **destination_data,
        "category_name": category_name
    }

# get destinations by id
@router.get("/{destination_id}", response_model=DestinationOut)
def get_destination_byId(destination_id: str):
    doc_ref = destination_collection.document(destination_id)
    destination = doc_ref.get()
    
    if not destination.exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    destination_data = destination.to_dict()
    destination_data['id'] = destination.id
    
    # Fetch category name
    category_id = destination_data.get('category_id')
    category_doc = category_collection.document(category_id).get()
    if category_doc.exists:
        destination_data['category_name'] = category_doc.to_dict().get('category_name')
    else:
        destination_data['category_name'] = "Unknown Category"

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
    
    category_id = category_doc.id

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
        "description": description,
        "destination_image": image_url,
        "image_phash": image_phash,
        "category_id": category_id
    }

    doc_ref.update(destination_data)
    
    return {
        "id": destination_id,
        **destination_data,
        "category_name": category_name
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
