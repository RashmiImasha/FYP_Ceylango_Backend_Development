from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
from app.routes.destination_route import destination_collection
from app.routes.category_route import collection
from app.models.image import ImageDescriptionRequest, ImageDescriptionResponse
from app.utils.gemini_analyzer import analyze_image_withAI, ImageAnalysis
import base64, uuid, io
import math, imagehash
from PIL import Image

router = APIRouter()

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

def map_category_name(gemini_category: str, existing_categories: list[str]) -> str:
    gemini_category = gemini_category.strip().lower()

    # Direct match
    for cat in existing_categories:
        if gemini_category == cat.lower():
            return cat

    # Match with keywords
    for valid_category, keywords in category_mapper.category_keywords_map.items():
        if any(keyword in gemini_category for keyword in keywords):
            if valid_category in existing_categories:
                return valid_category

    # Fallback
    return "General"

@router.post("/uploadImage", response_model=ImageAnalysis)
async def analyze_image(file: UploadFile = File(...)):
    try:
        # Read the image file and convert it to base64
        image_data = await file.read()
        encoded_image = base64.b64encode(image_data).decode('utf-8')

        # Call the AI analysis function
        prompt = (
        "Based on the image, identify the following:\n"
        "- The Location : \n"
        "- The district :\n"       
        "- Description including historical and cultural value."        
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
    # validate image type
    allowed_types = ["image/jpeg", "image/png"]
    if destination_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are allowed.")
    
    # read and encode image
    image_bytes = await destination_image.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

    # analyze image with AI
    prompt = (
        "Based on this image, identify the following:\n"
        "- The name of the place (landmark or natural location).\n"
        "- The district or local area name.\n"
        "- The type of place. Choose ONLY ONE from: Beach, Waterfalls, Mountains, Historical, Sacred, Rainforests, Gardens.\n"
        "- A short description including any historical or cultural value."
    )

    result = analyze_image_withAI(encoded_image, prompt,"snapImage")

    if not result:
        raise HTTPException(status_code=500, detail="Failed to analyze image with AI.")

    

    destination_name = result.destination_name
    district_name = result.district_name
    raw_category_name = result.raw_category_name
    description = result.description

    # category validation    
    category_query = collection.where('category_type', '==', 'location').stream()
    print("category_query",category_query  )
    available_categories = []
    category_doc_map = {}
    print("raw",raw_category_name)
    for doc in category_query:
        print("doc",doc)
        data = doc.to_dict()
        name = data['category_name']
        available_categories.append(name)
        category_doc_map[name] = doc

    # check existing destination
    existing_destinations = destination_collection.stream()
    for doc in existing_destinations:
        data = doc.to_dict()
        existing_lat = data.get("latitude")
        existing_lon = data.get("longitude")
        if existing_lat and existing_lon:
            distance = haversine(latitude, longitude, existing_lat, existing_lon)
            if distance < 5:
                raise HTTPException(status_code=400, detail="A destination already exists close to this location.")
            
    # # compute pHash and check existing image
    image_phash = compute_phash(io.BytesIO(image_bytes))
    for doc in existing_destinations:
        existing_phash = doc.to_dict().get('image_phash')
        if existing_phash:
            distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(image_phash)
            if distance <= 5:
                raise HTTPException(status_code=400, detail="A visually similar image already exists.")

    # upload image
    bucket = storage.bucket()
    image_id = str(uuid.uuid4())
    blob = bucket.blob(f'destination_images/{image_id}_{destination_image.filename}')
    blob.upload_from_string(image_bytes, content_type=destination_image.content_type)
    blob.make_public()
    image_url = blob.public_url

    # save to database
    destination_data = {
        "destination_name": destination_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_name": district_name,
        "description": description,
        "destination_image": image_url,
        "category_name": raw_category_name.value,
        "image_phash": image_phash
    }

    _, doc_ref = destination_collection.add(destination_data)
    print("destination",destination_data)
    return {
        "description": description             
    }







