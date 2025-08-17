from fastapi import HTTPException, UploadFile
from app.database.connection import category_collection, destination_collection
from app.utils.storage_handle import upload_file_to_storage, delete_file_from_storage
from PIL import Image
import imagehash, math
from typing import List, Optional

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
    upload_file.seek(0)
    image = Image.open(upload_file)
    
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    if image.width < 32 or image.height < 32:
        image = image.resize((32, 32))
    
    phash = imagehash.phash(image, hash_size=8)
    return str(phash)

def add_destination_record(destination_data: dict, images=None):
    """
    Adds a destination record to Firestore after applying all validation rules.
    return: dict with saved data
    """
    destination_name = destination_data["destination_name"]
    latitude = destination_data["latitude"]
    longitude = destination_data["longitude"]
    district_name = destination_data["district_name"]
    description = destination_data["description"]
    category_name = destination_data["category_name"]

    # Check if category exists
    category_query = category_collection \
        .where('category_name', '==', category_name) \
        .where('category_type', '==', 'location') \
        .get()
    if not category_query:
        raise HTTPException(status_code=404, detail="Category not found or not of type 'location'")

    # Check duplicate destination name
    existing_destination = destination_collection.where('destination_name', '==', destination_name).get()
    if existing_destination:
        raise HTTPException(status_code=400, detail="This Destination name already exists...!")

    # Check nearby (within 5m)
    for doc in destination_collection.stream():
        data = doc.to_dict()
        existing_lat = data.get("latitude")
        existing_lon = data.get("longitude")
        if existing_lat is not None and existing_lon is not None:
            distance = haversine(latitude, longitude, existing_lat, existing_lon)
            if distance < 5:
                raise HTTPException(status_code=400, detail="A destination already exists very close to this location")

    image_urls = []
    phash_list = []

    if images:  # Case: create_destination with UploadFile
        allow_types = ['image/jpeg', 'image/jpg', 'image/png']
        for img in images:
            if img.content_type not in allow_types:
                raise HTTPException(status_code=400, detail="Unsupported image file type")

        existing_phash_dicts = [doc.to_dict().get('image_phash', []) for doc in destination_collection.stream()]

        for img in images:
            img.file.seek(0)
            new_phash = compute_phash(img.file)

            for existing_phash_list in existing_phash_dicts:
                for existing_phash in existing_phash_list:
                    distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(new_phash)
                    if distance <= 5:
                        raise HTTPException(status_code=400, detail="A visually similar image already exists.")

            url = upload_file_to_storage(img, "destination_images")
            if url:
                image_urls.append(url)
                phash_list.append(new_phash)

    else:  # Case: move from missingplace (already has image + phash)
        image_urls = destination_data["destination_image"]
        phash_list = destination_data["image_phash"]

    # Save record
    record = {
        "destination_name": destination_name.strip(),
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name.strip(),
        "district_name_lower": district_name.strip().lower(),
        "description": description,
        "destination_image": image_urls,
        "image_phash": phash_list,
        "category_name": category_name.strip()
    }

    _, doc_ref = destination_collection.add(record)
    record["id"] = doc_ref.id
    return record

def update_destination_record(
    doc_ref,
    destination_id: str,
    collection,  
    destination_name: str,
    latitude: float,
    longitude: float,
    district_name: str,
    description: str,
    category_name: str,
    new_images: Optional[List[UploadFile]] = None,
    remove_existing: Optional[List[str]] = None
):
    """
    Shared update function for both destination and missingplace.
    """

    destination = doc_ref.get()
    if not destination.exists:
        raise HTTPException(status_code=404, detail="Record not found")

    current_data = destination.to_dict()
    current_data["id"] = destination_id

    # validate category
    category_query = category_collection \
        .where('category_name', '==', category_name) \
        .where('category_type', '==', 'location') \
        .get()
    if not category_query:
        raise HTTPException(status_code=404, detail="Valid location category not found")

    # duplicate name check
    existing_destination = collection.where('destination_name', '==', destination_name).stream()
    for doc in existing_destination:
        if doc.id != destination_id:
            raise HTTPException(status_code=400, detail="This Destination name already exists...!")

    # nearby check
    for doc in collection.stream():
        if doc.id == destination_id:
            continue
        data = doc.to_dict()
        if data.get("latitude") is not None and data.get("longitude") is not None:
            distance = haversine(latitude, longitude, data["latitude"], data["longitude"])
            if distance < 5:
                raise HTTPException(status_code=400, detail="Another destination is too close to this location")

    # handle images
    current_images = current_data.get("destination_image", []) or []
    current_phash = current_data.get("image_phash", []) or []

    if remove_existing:
        for url in remove_existing:
            if url in current_images:
                idx = current_images.index(url)
                current_images.pop(idx)
                current_phash.pop(idx)
                delete_file_from_storage({"destination_image": [url]}, {"destination_image": "destination_images"})

    if new_images:
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        existing_phash_lists = [
            doc.to_dict().get("image_phash", []) for doc in collection.stream() if doc.id != destination_id
        ]
        current_phash_set = set(current_phash)

        for img in new_images:
            if img.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail="Unsupported image file type")
            img.file.seek(0)
            phash = compute_phash(img.file)

            if phash in current_phash_set:
                raise HTTPException(status_code=400, detail="This image already exists in this Image List...!")

            for phash_list in existing_phash_lists:
                for existing_phash in phash_list:
                    if imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(phash) <= 5:
                        raise HTTPException(status_code=400, detail="A visually similar image already exists.")

            url = upload_file_to_storage(img, "destination_images")
            if url:
                current_images.append(url)
                current_phash.append(phash)
                current_phash_set.add(phash)

    if len(current_images) == 0:
        raise HTTPException(status_code=400, detail="At least one image must remain")

    # update data
    current_data.update({
        "destination_name": destination_name.strip(),
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name.strip(),
        "district_name_lower": district_name.strip().lower(),
        "description": description,
        "category_name": category_name.strip(),
        "destination_image": current_images,
        "image_phash": current_phash
    })

    doc_ref.set(current_data, merge=False)
    return {"id": destination_id, **current_data}