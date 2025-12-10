from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from firebase_admin import storage
from app.database.connection import misplace_collection, feedback_collection
from app.services.agent_service import get_agent_service
from app.services.pineconeService import get_pinecone_service
from app.models.image import SnapImageResponse
from app.utils.crud_utils import CrudUtils
from app.models.destination import MissingPlaceOut
from app.models.review import FeedbackRequest
from deep_translator import GoogleTranslator 
from gtts import gTTS
from PIL import Image
import uuid, io, logging, time, base64, asyncio
from typing import Optional, List
from datetime import datetime, timedelta
from google.cloud import firestore

logger = logging.getLogger(__name__)
router = APIRouter()

agent_service = get_agent_service()
pinecone_service = get_pinecone_service()

# ****************************************************
#  Content Generation routes
# ****************************************************

SUPPORTED_LANGUAGES = {
    "english": {"code": "en", "name": "English"},    
    "hindi": {"code": "hi", "name": "Hindi (हिन्दी)"},
    "chinese": {"code": "zh-CN", "name": "Chinese (中文)"},   
    "spanish": {"code": "es", "name": "Spanish (Español)"},
    "arabic": {"code": "ar", "name": "Arabic (العربية)"},
    "sinhala": {"code": "si", "name": "Sinhala (සිංහල)"},
    "tamil": {"code": "ta", "name": "Tamil (தமிழ்)"},
}

async def _save_missing_location(
    request_id: str,
    agent_result: dict,
    image_bytes: bytes,
    latitude: float,
    longitude: float,
    destination_image: UploadFile
):
    image_phash = CrudUtils.compute_phash(io.BytesIO(image_bytes))
    destination_name = agent_result.get('destination_name', 'Unknown')

    # Check for existing location by name 
    name_docs = list(
        misplace_collection
        .where("destination_name", "==", destination_name)
        .limit(1)
        .stream()
    )
    
    if name_docs:
        logger.info(f"[{request_id}] Skipped - Location already exists in missingplace collection")
        return
    
    # Check for duplicate by image hash
    phash_docs = list(
        misplace_collection
        .where("image_phash", "array_contains", image_phash)
        .limit(1)
        .stream()
    )
    
    if phash_docs:
        logger.info(f"[{request_id}] Skipped - Similar image already exists in missingplace collection")
        return

    # Build combined description
    fields = {
        "historical_background": "Historical Background",
        "cultural_significance": "Cultural Significance",
        "what_makes_it_special": "What Makes It Special",
        "visitor_experience": "Visitor Experience"
    }
    
    description_parts = [
        f"{title}: {agent_result[key]}"
        for key, title in fields.items()
        if agent_result.get(key)
    ]
    
    facts = agent_result.get("interesting_facts", [])
    if facts:
        description_parts.append("Interesting Facts: " + " | ".join(facts))
    
    combined_description = "\n\n".join(description_parts)

    # Prepare destination data
    destination_data = {
        "destination_name": destination_name,
        "latitude": latitude,
        "longitude": longitude,
        "district_name": agent_result.get('district_name', 'Unknown'),
        "description": combined_description,
        "destination_image": [],
        "category_name": agent_result.get('category', 'Others'),
        "image_phash": [image_phash],
        "created_at": firestore.SERVER_TIMESTAMP
    }

    # Upload image to Firebase Storage
    try:
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'missingplace_images/{image_id}_{destination_image.filename}')
        blob.upload_from_string(image_bytes, content_type=destination_image.content_type)
        blob.make_public()
        destination_data["destination_image"].append(blob.public_url)
    except Exception as upload_error:
        logger.error(f"[{request_id}] Image upload failed: {str(upload_error)}")
        # Continue without image
    
    # Save to Firestore
    _, doc_ref = misplace_collection.add(destination_data)
    logger.info(
        f"[{request_id}] Saved new location to missingplace: "
        f"{destination_name} (ID: {doc_ref.id})"
    )

@router.post("/snapImage", response_model=SnapImageResponse)
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
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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

        used_web_search = agent_result.get("used_web_search", False)
        if used_web_search:
            try:
                await _save_missing_location(
                    request_id=request_id,
                    agent_result=agent_result,
                    image_bytes=image_bytes,
                    latitude=latitude,
                    longitude=longitude,
                    destination_image=destination_image
                )
            except Exception as e:
                logger.error(f"[{request_id}] Failed to save missing location: {str(e)}")
           
        return {                   
                "destination_name": agent_result.get("destination_name", "Unknown"),
                "district_name": agent_result.get("district_name", "Unknown"),
                "historical_background": agent_result.get("historical_background", ""),
                "cultural_significance": agent_result.get("cultural_significance", ""),
                "what_makes_it_special": agent_result.get("what_makes_it_special", ""),
                "visitor_experience": agent_result.get("visitor_experience", ""),
                "interesting_facts": agent_result.get("interesting_facts", []),            
                "request_id": request_id                                     
            }             
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in snap_image_with_agent: {str(e)}", exc_info=True)
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
    request_id = str(uuid.uuid4())[:8]
    try:
        # Validation
        if not description.strip():
            raise HTTPException(status_code=400, detail="Description cannot be empty")
        
        if target_language not in SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported language. Choose from: {', '.join(SUPPORTED_LANGUAGES.keys())}"
            )

        lang_config = SUPPORTED_LANGUAGES[target_language]
        logger.info(f"[{request_id}] Target language: {lang_config['name']} ({lang_config['code']})")

        # Prepare base English text
        if include_intro and district_name:
            full_text_english = f"Information about {destination_name} in {district_name} district. {description}"
        else:
            full_text_english = description
        
        logger.info(f"[{request_id}] Full English text prepared (length: {len(full_text_english)} chars)")

        # Translation 
        if target_language != "english":
            logger.info(f"[{request_id}] Starting translation to {lang_config['name']}...")
            try:
                translator = GoogleTranslator(source="en", target=lang_config["code"])
                translated_text = await asyncio.to_thread(
                    translator.translate,
                    full_text_english
                )
                
            except Exception as trans_error:
                logger.error(f"[{request_id}] Translation failed: {str(trans_error)}")
                raise HTTPException(
                    status_code=500, 
                    detail=f"Translation failed: {str(trans_error)}"
                )

        else:
            logger.info(f"[{request_id}] No translation needed (target is English)")
            translated_text = full_text_english

        # TTS generation phase
        logger.info(f"[{request_id}] Starting TTS generation...")
        tts_start = time.time()
        
        try:
            def _tts_job():
                tts = gTTS(text=translated_text, lang=lang_config['code'], slow=slow)
                fp = io.BytesIO()
                tts.write_to_fp(fp)
                fp.seek(0)
                return fp

            audio_fp = await asyncio.to_thread(_tts_job)
            
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
        
        logger.info(f"[{request_id}] Generated audio stream successfully...!")

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

@router.post("/addFeedback")
def add_feedback(data: FeedbackRequest):
    try:
        record = {
            "feedback": data.feedback,
            "created_at": firestore.SERVER_TIMESTAMP
        }
        
        _, doc_ref = feedback_collection.add(record)
        record["id"] = doc_ref.id

        logger.info(f"Feedback submitted successfully...! ID: {record['id']}")
        return {
            "message": "Feedback submitted successfully...!",
            "data": record["feedback"]
        }

    except Exception as e:
        logger.error(f"Error submitting feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/feedbackAnalytics")
async def get_feedback_analytics(days: int = 30):
    try:
        # Time window
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Fetch all feedback
        docs = (
            feedback_collection
            .where("created_at", ">=", start_date)
            .where("created_at", "<=", end_date)
            .stream()
        )

        # Base counters
        distribution = {
            "excellent": 0,
            "good": 0,
            "acceptable": 0,
            "poor": 0,
            "incorrect": 0
        }

        timeseries = {}

        for doc in docs:
            data = doc.to_dict()
            fb = data.get("feedback")
            created_at = data.get("created_at")

            # Update distribution
            if fb in distribution:
                distribution[fb] += 1

            # Update timeseries
            if created_at:
                if hasattr(created_at, "to_datetime"):
                    created_at = created_at.to_datetime()

                date_key = created_at.strftime("%Y-%m-%d")

                if date_key not in timeseries:
                    timeseries[date_key] = {
                        "excellent": 0,
                        "good": 0,
                        "acceptable": 0,
                        "poor": 0,
                        "incorrect": 0
                    }

                if fb in timeseries[date_key]:
                    timeseries[date_key][fb] += 1
            
        total_count = sum(distribution.values())

        logger.info("Feedback analytics fetched successfully")
        return {
            "distribution": distribution,
            "timeseries": timeseries,
            "total_count": total_count
        }

    except Exception as e:
        logger.error(f"Error fetching feedback analytics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ****************************************************
#  missing-place management routes
# ****************************************************

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
            local_paths = CrudUtils.move_files_to_new_folder(
                urls=data["destination_image"],
                source_folder="missingplace_images",
                target_folder="destination_images"
            )

            # Upload the files to GCS (or storage) to get accessible URLs
            new_image_urls = []
            for local_path in local_paths:
                uploaded_url = CrudUtils.upload_file_to_storage(local_path, "destination_images")
                new_image_urls.append(uploaded_url)
            
            # Update data with uploaded URLs
            data["destination_image"] = new_image_urls
            

        # Add to destination collection (which now syncs to Pinecone)
        record = CrudUtils.add_destination_record(data, images=None)
        logger.info(f"Moved missing place to destination collection : ID {record['id']}")
        
        # Sync to Pinecone
        pinecone_service.upsert_destination_image(record['id'], record)
        logger.info(f"Synced record to Pinecone: ID {record['id']}")
        
        # Delete from missingplace
        doc_ref.delete()
        logger.info(f"Deleted from missing place collection: ID {missingplace_id}")
        
        return {
            "message": "Moved to destination and synced to Pinecone",
            "destination_id": record["id"]
        }
    
    except Exception as e:
        logger.error(f"Error moving to destination: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{missing_id}")
def get_missing(missing_id: str):
    return CrudUtils.get_by_id(misplace_collection, missing_id)

@router.get("/")
def get_all_missing():
    return CrudUtils.get_all(misplace_collection)

@router.delete("/{missing_id}")
def delete_missing(missing_id: str):
    files_mapping = {
        "destination_image": "missingplace_images",
    }
    return CrudUtils.delete_by_id(misplace_collection, missing_id, files_mapping)

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
    return CrudUtils.update_destination_record(
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



