from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
from app.database.connection import destination_collection, misplace_collection
from app.utils.google_analyzer import analyze_image_withAI, ImageAnalysis
from app.utils.destinationUtils import haversine, compute_phash, add_destination_record, update_destination_record
from app.utils.storage_handle import delete_file_from_storage, move_files_to_new_folder
import base64, uuid, io, imagehash
from app.utils.crud_utils import get_all, get_by_id, delete_by_id
from app.models.destination import MissingPlaceOut
from typing import Optional, List

router = APIRouter()

@router.post("/uploadImage", response_model=ImageAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_data = await file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Call the AI analysis function
        prompt = (
             "You are a Sri Lankan heritage expert. Given the image, provide:\n"
                "1. The exact place name.\n"
                "2. Its district.\n"
                "3. A brief but informative historical and cultural description.\n"
                "Note: Avoid guessing. Respond only with confirmed facts visible in the image."                        
        )

        result = analyze_image_withAI(encoded_image, prompt,"uploadImage")
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the image: {str(e)}")
    
@router.post("/snapImage")
async def snap_image_analyze(
    latitude: float = Form(...),
    longitude: float = Form(...),
    destination_image: UploadFile = File(...),  
):    
    # read and encode image
    image_bytes = await destination_image.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')
    image_phash = compute_phash(io.BytesIO(image_bytes))

    # check nearby location
    nearby_destinations = []
    for doc in destination_collection.stream():
        data = doc.to_dict()
        if data.get("latitude") is not None and data.get("longitude") is not None:
            distance = haversine(latitude, longitude, data["latitude"], data["longitude"])
            if distance <= 10000:  # 10 km
                data['id'] = doc.id
                nearby_destinations.append(data)
    
    # if nearby loc found, check for similar images
    assumed_district = "Unknown"
    if nearby_destinations:
        for dest in nearby_destinations:
            for existing_phash in dest.get("image_phash", []):
                phash_distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(image_phash)
                if phash_distance <= 5:
                    # Similar image found, return destination info
                    return {
                        "destination_name": dest.get("destination_name"),
                        "district_name": dest.get("district_name"),
                        "description": dest.get("description")
                    }
    
        # Nearby found, but no similar image: (assume district name)
        closest_dest = min(
                nearby_destinations,
                key=lambda d: haversine(latitude, longitude, d["latitude"], d["longitude"])
        )
        assumed_district = closest_dest.get("district_name", "Unknown")

        prompt = (
                f"You are analyzing a location based on an image and coordinates.\n"
                f"Latitude: {latitude}, Longitude: {longitude}\n"
                f"This location is within {assumed_district} district.\n\n"
                "Use BOTH the image and this district context to identify the place.\n"
                "If confident, return:\n"
                "- destination_name\n"
                "- district_name\n"
                "- type (Beach, Waterfalls, Mountains, Historical, Sacred, Rainforests, Gardens)\n"
                "- description\n\n"
                "If you cannot match it even with the district context, return Unknown fields."
            )

        result = analyze_image_withAI(encoded_image, prompt, "snapImage")
        if not result:
            raise HTTPException(status_code=500, detail="Failed to analyze image with AI.")

        # Save to missingplace collection
        destination_data = {
            "destination_name": result.destination_name,
            "latitude": latitude,
            "longitude": longitude,
            "district_name": result.district_name if result.district_name != "Unknown" else assumed_district,
            "description": result.description,
            "destination_image": [],
            "category_name": result.raw_category_name.value if result.raw_category_name else "Unknown",
            "image_phash": [image_phash]
        }

        # Upload image
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'missingplace_images/{image_id}_{destination_image.filename}')
        blob.upload_from_string(image_bytes, content_type=destination_image.content_type)
        blob.make_public()
        destination_data["destination_image"].append(blob.public_url)

        _, doc_ref = misplace_collection.add(destination_data)

        return {
            "destination_name": destination_data["destination_name"],
            "district_name": destination_data["district_name"],
            "description": destination_data["description"],
            # "status": "Saved in missingplace collection"
        }
    
    else:
        # analyze image with AI
        prompt = (
            f"You are analyzing a location based on an image and coordinates.\n"
            f"The image was taken at the following coordinates:\n"
            f"- Latitude: {latitude}\n"
            f"- Longitude: {longitude}\n\n"
            "Your task is to identify the location *only if* both the image content and coordinates strongly support the same place.\n"
            "If there is any mismatch, uncertainty, or if the location cannot be confidently determined, respond with the following:\n"
            '- destination_name: "Unknown"\n'
            '- district_name: "Unknown"\n'
            '- type: "Unknown"\n'
            '- description: "Could not determine based on available data."\n\n'

            "If the image and coordinates clearly indicate a known place, respond with:\n"
            "1. The exact name of the place (landmark or natural location).\n"
            "2. The district or local area name.\n"
            "3. The type of place. Choose ONLY ONE from: Beach, Waterfalls, Mountains, Historical, Sacred, Rainforests, Gardens.\n"
            "4. A brief but informative historical and cultural description. ( 4 sentences)\n\n"
            " Be extremely accurate and give the answer *only* if the image and coordinates clearly match a known location."
        )

        result = analyze_image_withAI(encoded_image, prompt,"snapImage")
        if not result:
            raise HTTPException(status_code=500, detail="Failed to analyze image with AI.")  
        
        # Save to missingplace collection
        destination_data = {
            "destination_name": result.destination_name,
            "latitude": latitude,
            "longitude": longitude,
            "district_name": result.district_name if result.district_name != "Unknown" else assumed_district,
            "description": result.description,
            "destination_image": [],
            "category_name": result.raw_category_name.value if result.raw_category_name else "Unknown",
            "image_phash": [image_phash]
        }

        # Upload image
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'missingplace_images/{image_id}_{destination_image.filename}')
        blob.upload_from_string(image_bytes)
        blob.make_public()
        destination_data["destination_image"].append(blob.public_url)

        _, doc_ref = misplace_collection.add(destination_data)

        return {
        "destination_name": destination_data["destination_name"],
        "district_name": destination_data["district_name"],
        "description": destination_data["description"],
        # "status": "Saved in missingplace"
        }

    # destination_name = result.destination_name
    # district_name = result.district_name
    # raw_category_name = result.raw_category_name
    # description = result.description

    # # category validation    
    # category_query = collection.where('category_type', '==', 'location').stream()
    # print("category_query",category_query  )
    # available_categories = []
    # category_doc_map = {}
    # print("raw",raw_category_name)
    # for doc in category_query:
    #     print("doc",doc)
    #     data = doc.to_dict()
    #     name = data['category_name']
    #     available_categories.append(name)
    #     category_doc_map[name] = doc

    # # check existing destination
    # duplicate_found = False
    # existing_destinations = destination_collection.stream()
    # for doc in existing_destinations:
    #     data = doc.to_dict()
    #     existing_lat = data.get("latitude")
    #     existing_lon = data.get("longitude")
    #     if existing_lat and existing_lon:
    #         distance = haversine(latitude, longitude, existing_lat, existing_lon)
    #         if distance < 5:
    #             duplicate_found = True
    #             break
    #             # raise HTTPException(status_code=400, detail="A destination already exists close to this location.")
            
    # # compute pHash and check existing image
    # image_phash = compute_phash(io.BytesIO(image_bytes))
    # for doc in existing_destinations:
    #     existing_phash = doc.to_dict().get('image_phash')
    #     if existing_phash:
    #         distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(image_phash)
    #         if distance <= 5:
    #             duplicate_found = True
    #             break
    #             # raise HTTPException(status_code=400, detail="A visually similar image already exists.")
    
    # if not duplicate_found:
    #     # upload image
    #     bucket = storage.bucket()
    #     image_id = str(uuid.uuid4())
    #     blob = bucket.blob(f'destination_images/{image_id}_{destination_image.filename}')
    #     blob.upload_from_string(image_bytes, content_type=destination_image.content_type)
    #     blob.make_public()
    #     image_url = blob.public_url

    #     # save to database
    #     destination_data = {
    #         "destination_name": destination_name,
    #         "latitude": latitude,
    #         "longitude": longitude,
    #         "district_name": district_name,
    #         "description": description,
    #         "destination_image": image_url,
    #         "category_name": raw_category_name.value,
    #         "image_phash": image_phash,
    #         "district_name_lower": district_name.lower(),
    #     }

    #     _, doc_ref = destination_collection.add(destination_data)
    #     print("destination",destination_data)
        
    # return {
    #     "destination_name": destination_name,
    #     "district_name": district_name,
    #     "description": description             
    # }

@router.post("/moveToDestination")
def move_missing_to_destination(missingplace_id: str):
    doc_ref = misplace_collection.document(missingplace_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Missing place entry not found.")

    data = doc.to_dict()

    try:
        if "destination_image" in data and data["destination_image"]:
            new_image_urls = move_files_to_new_folder(
                urls=data["destination_image"],
                source_folder="missingplace_images",
                target_folder="destination_images"
            )
            data["destination_image"] = new_image_urls

        record = add_destination_record(data, images=None)  # images=None → use existing image URLs + phash
        # files_mapping = {"destination_image": "missingplace_images"}  # adjust field and folder name
        # delete_file_from_storage(data, files_mapping)
        
        doc_ref.delete()
        return {"message": "Moved to destination successfully", "destination_id": record["id"]}
    
    except Exception as e:
        return {"message": "Failed to move to destination", "error": str(e)}

@router.get("/{missing_id}")
def get_missing(missing_id: str):
    return get_by_id(misplace_collection, missing_id)

@router.get("/")
def get_all_missing():
    return get_all(misplace_collection)

@router.delete("/{missing_id}")
def delete_missing(missing_id: str):
    files_mapping = {
        "destination_image": "missingplace_images",
    }
    return delete_by_id(misplace_collection, missing_id, files_mapping)

@router.put("/{missing_id}", response_model=MissingPlaceOut)
def update_misplace(
    misplace_id: str,
    destination_name: str = Form(...),
    latitude: float = Form(...),
    longitude: float = Form(...),
    district_name: str = Form(...),
    description: str = Form(...),
    category_name: str = Form(...),
    new_images: Optional[List[UploadFile]] = File(None),
    remove_existing: Optional[List[str]] = Form(None)
):
    doc_ref = misplace_collection.document(misplace_id)
    return update_destination_record(
        doc_ref=doc_ref,
        destination_id=misplace_id,
        collection=misplace_collection,   
        destination_name=destination_name,
        latitude=latitude,
        longitude=longitude,
        district_name=district_name,
        description=description,
        category_name=category_name,
        new_images=new_images,
        remove_existing=remove_existing
    )





