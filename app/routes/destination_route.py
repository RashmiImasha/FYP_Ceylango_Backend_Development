from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
from app.database.connection import db
import uuid
import hashlib

from app.models.destination import Destination, DestinationOut
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
    category_id: str = Form(...)
):
    # Check if category exists
    category_doc = category_collection.document(category_id).get()
    if not category_doc.exists:
        raise HTTPException(status_code=404, detail="Category not found")
    
    category_name = category_doc.to_dict().get('category_name')

    # check if destination name already exists
    existing_destination = destination_collection.where('destination_name', '==', destination_name).get()
    if existing_destination:
        raise HTTPException(status_code=400, detail="This Destination name already exists...!")

    # Check if destination is already exists at this lat + lng
    lat_matches = destination_collection.where('latitude', '==', latitude).stream()
    for doc in lat_matches:
        data = doc.to_dict()
        # longitude comparison
        if abs(data['longitude'] - longitude) < 1e-6:
            raise HTTPException(status_code=400, detail="Destination at this latitude and longitude already exists")

    # Compute SHA-256 hash of the image content
    image_bytes = destination_image.file.read()
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    # Check if image hash already exists
    existing_dest_image = destination_collection.stream()
    for doc in existing_dest_image:
        if doc.to_dict().get('image_hash') == image_hash:
            raise HTTPException(status_code=400, detail="Image with this hash already exists")
    
    # upload image to firebase storage
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
        "image_hash": image_hash,
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
    destination_data['category_name'] = category_doc.to_dict().get('category_name') if category_doc.exists else "Unknown Category"

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
    destination_image: UploadFile = File(...),
    category_id: str = Form(...)
):
    doc_ref = destination_collection.document(destination_id)
    destination = doc_ref.get()
    
    if not destination.exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    current_data = destination.to_dict()
    
    # Check if category exists
    category_doc = category_collection.document(category_id).get()
    if not category_doc.exists:
        raise HTTPException(status_code=404, detail="Category not found")
    
    category_name = category_doc.to_dict().get('category_name')

    # Check if destination name already exists
    existing_destination = destination_collection.where('destination_name', '==', destination_name).stream()
    if existing_destination and existing_destination[0].id != destination_id:
        raise HTTPException(status_code=400, detail="This Destination name already exists...!")
    
    # Check for duplicate latitude + longitude (excluding self)
    lat_matches = destination_collection.where("latitude", "==", latitude).stream()
    for doc in lat_matches:
        data = doc.to_dict()
        if doc.id != destination_id and abs(data.get("longitude") - longitude) < 1e-6:
            raise HTTPException(status_code=400, detail="A destination with this latitude and longitude already exists.")

    # upload image to firebase storage
    image_url = current_data.get("destination_image")
    image_hash = current_data.get("image_hash")

    if destination_image:
        # Compute SHA-256 hash of the new image content
        new_image_bytes = destination_image.file.read()
        new_image_hash = hashlib.sha256(new_image_bytes).hexdigest()

        # Check if image hash already exists
        existing_dest_image = destination_collection.stream()
        for doc in existing_dest_image:
            if doc.to_dict().get('image_hash') == new_image_hash and doc.id != destination_id:
                raise HTTPException(status_code=400, detail="Image with this hash already exists")
        
        image_hash = new_image_hash
        
        # Delete old image
        if image_url:
            parsed_url = urlparse(image_url)
            image_path = parsed_url.path.lstrip('/')
            blob = storage.bucket().blob(image_path)
            if blob.exists():
                blob.delete()

        # Upload new image
        destination_image.file.seek(0)  # Reset file pointer to the beginning
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
        "image_hash": image_hash,
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
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Destination not found")
    
    data = doc_ref.to_dict()

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
