from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from firebase_admin import storage
from app.database.connection import misplace_collection
from app.services.agent import get_agent_service
from app.services.pineconeService import get_pinecone_service
from app.utils.destinationUtils import compute_phash
from app.utils.storage_handle import move_files_to_new_folder
from app.utils.crud_utils import get_all, get_by_id, delete_by_id
from app.models.destination import MissingPlaceOut
from deep_translator import GoogleTranslator 
from gtts import gTTS
from PIL import Image
import uuid, io, logging, time, base64
from typing import Optional, List


logger = logging.getLogger(__name__)
router = APIRouter()

agent_service = get_agent_service()
pinecone_service = get_pinecone_service()
translator = GoogleTranslator()

SUPPORTED_LANGUAGES = {
    "english": {"code": "en", "name": "English"},
    "sinhala": {"code": "si", "name": "Sinhala (සිංහල)"},
    "tamil": {"code": "ta", "name": "Tamil (தமிழ்)"},
    "hindi": {"code": "hi", "name": "Hindi (हिन्दी)"},
    "japanese": {"code": "ja", "name": "Japanese (日本語)"},
    "chinese": {"code": "zh-CN", "name": "Chinese (中文)"},
    "french": {"code": "fr", "name": "French (Français)"},
    "german": {"code": "de", "name": "German (Deutsch)"},
    "spanish": {"code": "es", "name": "Spanish (Español)"},
    "korean": {"code": "ko", "name": "Korean (한국어)"}
}

# @router.post("/uploadImage")
# async def analyze_image(file: UploadFile = File(...)):
#     try:
#         # Read and process image
#         image_bytes = await file.read()
#         image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#         # Use agent for identification (no GPS)
#         gps_location = {"lat": None, "lng": None}
#         agent_result = await agent_service.identify_and_generate_content(
#             image=image,
#             gps_location=gps_location
#         )

#         if not agent_result.get("success"):
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Agent failed: {agent_result.get('error')}"
#             )

#         return agent_result

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"An error occurred while processing the image: {str(e)}"
#         )

@router.post("/snapImage")
async def snap_image_with_agent(
    latitude: float = Form(...),
    longitude: float = Form(...),
    destination_image: UploadFile = File(...),    
):   
    request_id = str(uuid.uuid4())[:8]

    try:
        # Read image
        image_bytes = await destination_image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = image.convert('RGB')
            logger.info(f"Image opened successfully: size={image.size}, mode={image.mode}")
        except Exception as e:
            logger.error(f"Failed to open image: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
            
        gps_location = {'lat': latitude, 'lng': longitude}

        # Call agent
        agent_result = await agent_service.identify_and_generate_content(
            image=image,
            gps_location=gps_location,
        )

        if not agent_result.get("success"):
            raise HTTPException(status_code=500, detail=f"Agent failed: {agent_result.get('error')}")    

        found_in_db = agent_result.get("found_in_db", False)
        used_web_search = agent_result.get("used_web_search", False)  

        if not found_in_db or used_web_search:

            image_phash = compute_phash(io.BytesIO(image_bytes))

            # Save to missingplace collection for admin review
            destination_data = {
                "destination_name": agent_result.get('destination_name', 'Unknown'),
                "latitude": latitude,
                "longitude": longitude,
                "district_name": agent_result.get('district_name', 'Unknown'),
                "description": agent_result.get('description', ''),
                "destination_image": [],
                "category_name": agent_result.get('category', 'Others'),
                "image_phash": [image_phash]
            }

            existing_docs = misplace_collection.stream()
            already_exists = False
            existing_data = None
            for doc in existing_docs:
                data = doc.to_dict()
                if (
                    destination_data["image_phash"][0] in data.get("image_phash", [])
                    or data.get("destination_name", "").strip().lower() == destination_data["destination_name"].strip().lower()
                ):
                    already_exists = True
                    existing_data = data
                    break

            if already_exists:
                logger.info(f"Duplicate entry found in missingplace: {existing_data.get('destination_name', 'Unknown')}")                
                return {                   
                    "destination_name": destination_data["destination_name"],
                    "district_name": destination_data["district_name"],            
                    "description": destination_data["description"],                                        
                }                

            # Upload new image to missingplace storage 
            bucket = storage.bucket()
            image_id = str(uuid.uuid4())
            blob = bucket.blob(f'missingplace_images/{image_id}_{destination_image.filename}')
            blob.upload_from_string(image_bytes, content_type=destination_image.content_type)
            blob.make_public()
            destination_data["destination_image"].append(blob.public_url)       
            
            # Save to Firebase
            _, doc_ref = misplace_collection.add(destination_data)
            logger.info("Location not in database - Saving to missing-place collection")
            
            return {                                                  
                "destination_name": destination_data["destination_name"],
                "district_name": destination_data["district_name"],            
                "description": destination_data["description"],      
                "request_id": request_id                  
            }
        
        else:
            logger.info(f"Location already in database: {agent_result.get('destination_name')}")
            return {                   
                    "destination_name": agent_result.get("destination_name", "Unknown"),
                    "district_name": agent_result.get("district_name", "Unknown"),            
                    "description": agent_result.get("description", ""),   
                    "request_id": request_id                                     
                }             
        
    except Exception as e:
        logger.error(f"Error in snap_image_with_agent: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process image: {str(e)}"
        )

@router.get("/getLanguage")
async def get_supported_languages():

    languages = [
        {"key": key, "code": config["code"], "name": config["name"]}
        for key, config in SUPPORTED_LANGUAGES.items()
    ]
    
    return {
        "success": True,
        "languages": languages,
        "total": len(languages)
    }

@router.post("/translateAndSpeak")
async def translate_and_speak(
    destination_name: str = Form(...),
    description: str = Form(...),
    district_name: str = Form(default=""),
    target_language: str = Form(default="english"),
    include_intro: bool = Form(default=True),
    slow: bool = Form(default=False),
):
    """
    Translate text and generate audio in one step
    """   
    request_id = str(uuid.uuid4())[:8]

    try:
        logger.info(f"[{request_id}] translateAndSpeak request received")
        logger.info(f"[{request_id}] Parameters: destination_name='{destination_name}', "
                   f"district_name='{district_name}', target_language='{target_language}', "
                   f"include_intro={include_intro}, slow={slow}")
        logger.info(f"[{request_id}] Description length: {len(description)} chars")
        logger.info(f"[{request_id}] Description preview: {description[:100]}...")

        # Validation
        if not description.strip():
            logger.warning(f"[{request_id}] Empty description received")
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        
        if target_language not in SUPPORTED_LANGUAGES:
            logger.warning(f"[{request_id}] Unsupported language requested: {target_language}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )

        lang_config = SUPPORTED_LANGUAGES[target_language]
        logger.info(f"[{request_id}] Target language: {lang_config['name']} ({lang_config['code']})")

        # Prepare base English text
        if include_intro:
            if district_name and district_name != "Unknown":
                full_text_english = f"Information about {destination_name} in {district_name} district. {description}"
            else:
                full_text_english = f"Information about {destination_name}. {description}"
        else:
            full_text_english = description
        
        logger.info(f"[{request_id}] Full English text prepared (length: {len(full_text_english)} chars)")

        # Translation phase
        translated_text = full_text_english
        if target_language != "english":
            logger.info(f"[{request_id}] Starting translation to {lang_config['name']}...")
            translation_start = time.time()
            
            try:
                translation = translator.translate(
                    full_text_english, dest=lang_config['code'], src='en'
                )
                translated_text = translation.text
                
                translation_time = time.time() - translation_start
                logger.info(f"[{request_id}] Translation completed in {translation_time:.2f}s")
                logger.info(f"[{request_id}] Translated text length: {len(translated_text)} chars")
                
            except Exception as trans_error:
                logger.error(f"[{request_id}] Translation failed: {str(trans_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Translation failed: {str(trans_error)}"
                )
        else:
            logger.info(f"[{request_id}] No translation needed (target is English)")

        # TTS generation phase
        logger.info(f"[{request_id}] Starting TTS generation...")
        tts_start = time.time()
        
        try:
            tts = gTTS(text=translated_text, lang=lang_config['code'], slow=slow)
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            audio_fp.seek(0)
            
            audio_size = audio_fp.getbuffer().nbytes
            tts_time = time.time() - tts_start
            
            logger.info(f"[{request_id}] TTS generation completed in {tts_time:.2f}s")
            logger.info(f"[{request_id}] Audio file size: {audio_size / 1024:.2f} KB")
            
        except Exception as tts_error:
            logger.error(f"[{request_id}] TTS generation failed: {str(tts_error)}")
            raise HTTPException(
                status_code=500, 
                detail=f"TTS generation failed: {str(tts_error)}"
            )

        # Encode translated text for header (base64 to handle Unicode characters)
        translated_text_encoded = base64.b64encode(translated_text.encode('utf-8')).decode('ascii')
        
        # Sanitize filename to avoid issues with special characters
        safe_filename = "".join(c for c in destination_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_filename = safe_filename.replace(' ', '_') or 'destination'
        
        logger.info(f"[{request_id}] Request completed successfully - Returning audio stream")

        # Return audio with metadata
        return StreamingResponse(
            audio_fp,
            media_type="audio/mpeg",
            headers={
                "X-Translated-Text": translated_text_encoded,  
                "X-Text-Encoding": "base64", 
                "X-Request-ID": request_id,
                "Content-Disposition": f'inline; filename="{safe_filename}_{target_language}.mp3"'
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error in translateAndSpeak: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ********missing-place management routes********

@router.post("/moveToDestination")
def move_missing_to_destination(missingplace_id: str):
    """Move from missingplace to destination collection and sync to Pinecone"""
    
    doc_ref = misplace_collection.document(missingplace_id)
    doc = doc_ref.get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Missing place entry not found.")

    data = doc.to_dict()

    try:
        # Move images to destination folder
        if "destination_image" in data and data["destination_image"]:
            new_image_urls = move_files_to_new_folder(
                urls=data["destination_image"],
                source_folder="missingplace_images",
                target_folder="destination_images"
            )
            data["destination_image"] = new_image_urls

        # Add to destination collection (which now syncs to Pinecone)
        from app.utils.destinationUtils import add_destination_record
        record = add_destination_record(data, images=None)
        
        # Sync to Pinecone
        pinecone_service.upsert_destination(record['id'], record)
        
        # Delete from missingplace
        doc_ref.delete()
        
        return {
            "message": "Moved to destination and synced to Pinecone",
            "destination_id": record["id"]
        }
    
    except Exception as e:
        logger.error(f"Error moving to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
    """Update missingplace entry"""
    from app.utils.destinationUtils import update_destination_record
    
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



