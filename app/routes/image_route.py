from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
from app.routes.destination_route import destination_collection
from app.routes.category_route import collection
from app.utils.google_analyzer import analyze_image_withAI, ImageAnalysis
from app.utils.destinationUtils import haversine, compute_phash
import base64, uuid, io, imagehash

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
    # validate image type
    allowed_types = ["image/jpeg", "image/png"]
    if destination_image.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Invalid image type. Only JPEG and PNG are allowed.")
    
    # read and encode image
    image_bytes = await destination_image.read()
    encoded_image = base64.b64encode(image_bytes).decode('utf-8')

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
        "4. A brief but informative historical and cultural description.\n\n"
        " Be extremely accurate and give the answer *only* if the image and coordinates clearly match a known location."
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
    duplicate_found = False
    existing_destinations = destination_collection.stream()
    for doc in existing_destinations:
        data = doc.to_dict()
        existing_lat = data.get("latitude")
        existing_lon = data.get("longitude")
        if existing_lat and existing_lon:
            distance = haversine(latitude, longitude, existing_lat, existing_lon)
            if distance < 5:
                duplicate_found = True
                break
                # raise HTTPException(status_code=400, detail="A destination already exists close to this location.")
            
    # compute pHash and check existing image
    image_phash = compute_phash(io.BytesIO(image_bytes))
    for doc in existing_destinations:
        existing_phash = doc.to_dict().get('image_phash')
        if existing_phash:
            distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(image_phash)
            if distance <= 5:
                duplicate_found = True
                break
                # raise HTTPException(status_code=400, detail="A visually similar image already exists.")
    
    if not duplicate_found:
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
            "image_phash": image_phash,
            "district_name_lower": district_name.lower(),
        }

        _, doc_ref = destination_collection.add(destination_data)
        print("destination",destination_data)
        
    return {
        "destination_name": destination_name,
        "district_name": district_name,
        "description": description             
    }







