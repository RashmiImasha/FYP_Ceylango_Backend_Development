from typing import Optional, Dict, Any, List
import uuid
import json
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, Query
from firebase_admin import auth
from app.database.connection import db
from app.models.user import (
    UpdateProfileRequest,
   
)
from app.utils.storage_handle import upload_file_to_storage, delete_file_from_storage
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()
profiles_collection = db.collection("service_provider_profiles")
users_collection = db.collection("users")
security = HTTPBearer()



class PosterMetadata(BaseModel):
    name: str
    description: str
    expiration_date: str


class UpdatePosterMetadataRequest(BaseModel):
    poster_id: str
    name: str
    description: str
    expiration_date: str

class DeleteImagesRequest(BaseModel):
    image_urls: List[str]


# ===== AUTH DEPENDENCY =====
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        
        user_doc = users_collection.document(uid).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        if user_data.get("role") != "service_provider":
            raise HTTPException(status_code=403, detail="Access denied. Service provider role required")
        
        return uid, user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


# ===== ENDPOINTS =====

@router.get("/profile")
async def get_my_profile(current_user: tuple = Depends(get_current_user)):
    """Get service provider's own profile"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid).get()
        
        if not profile_doc.exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile_doc.to_dict()
        profile_data["uid"] = uid
        
        return {
            "profile": profile_data,
            "user_info": {
                "email": user_data.get("email"),
                "full_name": user_data.get("full_name"),
                "status": user_data.get("status")
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.put("/profile")
async def update_profile(
    request: UpdateProfileRequest,
    current_user: tuple = Depends(get_current_user)
):
    """Update all profile information in a single request"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        
        profile_snapshot = profile_doc.get()
        if not profile_snapshot.exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Get existing profile data to preserve service_category
        existing_profile = profile_snapshot.to_dict()
        
        # Prepare complete update data
        update_data = {
            # Basic Info
            "service_name": request.service_name.strip(),
            "description": request.description.strip(),
            "address": request.address.strip(),
            "district": request.district.strip(),
            "phone_number": request.phone_number.strip(),
            
            # Preserve service_category from existing profile
            "service_category": existing_profile.get("service_category"),
            
            # Location
            "coordinates": {
                "lat": request.latitude, 
                "lng": request.longitude
            },
            
            # Operating Hours - already in correct format from frontend
            "operating_hours": request.operating_hours,
            
            # Social Media
            "social_media": request.social_media,
            
            # Amenities
            "amenities": request.amenities,
            
            # Timestamp
            "updated_at": datetime.now().isoformat()
        }
        
        # Add optional fields
        if request.email:
            update_data["email"] = request.email.strip()
        if request.website:
            update_data["website"] = request.website.strip()
        
        # Single update operation
        profile_doc.update(update_data)
        
        return {
            "message": "Profile updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ✅ FormData only for file uploads
@router.post("/profile/images/profile")
async def upload_profile_images(
    images: List[UploadFile] = File(...),
    current_user: tuple = Depends(get_current_user)
):
    """Upload profile images (regular service images)"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        for img in images:
            if img.content_type not in allowed_types:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {img.content_type}"
                )
        
        # Upload images
        image_urls = []
        for img in images:
            url = upload_file_to_storage(img, "service_provider_images")
            if url:
                image_urls.append(url)
        
        # Get current images and append new ones
        current_images = profile_data.get("profile_images", [])
        current_images.extend(image_urls)
        
        profile_doc.update({
            "profile_images": current_images,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": f"{len(image_urls)} images uploaded successfully",
            "uploaded_images": image_urls,
            "total_images": len(current_images)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ✅ FormData for file + metadata as JSON string
@router.post("/profile/images/posters")
async def upload_poster_image(
    poster: UploadFile = File(...),
    metadata: str = Form(...),  # JSON string
    current_user: tuple = Depends(get_current_user)
):
    """Upload a promotional poster with metadata"""
    uid, user_data = current_user
    
    try:
        # Parse JSON metadata
        poster_metadata = PosterMetadata.parse_raw(metadata)
        
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Validate file type
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
        if poster.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {poster.content_type}"
            )
        
        # Upload poster image
        image_url = upload_file_to_storage(poster, "service_provider_posters")
        
        if not image_url:
            raise HTTPException(status_code=500, detail="Failed to upload image")
        
        # Create poster object
        poster_id = str(uuid.uuid4())
        poster_data = {
            "id": poster_id,
            "name": poster_metadata.name.strip(),
            "description": poster_metadata.description.strip(),
            "expiration_date": poster_metadata.expiration_date,
            "image_url": image_url,
            "created_at": datetime.now().isoformat()
        }
        
        current_posters = profile_data.get("poster_images", [])
        current_posters.append(poster_data)
        
        profile_doc.update({
            "poster_images": current_posters,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Poster uploaded successfully",
            "poster": poster_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/images/posters/metadata")
async def update_poster_metadata(
    request: UpdatePosterMetadataRequest,
    current_user: tuple = Depends(get_current_user)
):
    """Update poster metadata (name, description, expiration_date) without changing the image"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        current_posters = profile_data.get("poster_images", [])
        
        # Find and update the specific poster
        poster_found = False
        updated_posters = []
        
        for poster in current_posters:
            if poster.get("id") == request.poster_id:
                poster_found = True
                # Update only the metadata fields
                poster["name"] = request.name.strip()
                poster["description"] = request.description.strip()
                poster["expiration_date"] = request.expiration_date
                poster["updated_at"] = datetime.now().isoformat()
            updated_posters.append(poster)
        
        if not poster_found:
            raise HTTPException(status_code=404, detail="Poster not found")
        
        # Update the posters array in Firestore
        profile_doc.update({
            "poster_images": updated_posters,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Poster metadata updated successfully",
            "poster_id": request.poster_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error updating poster metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ✅ JSON endpoint for deletions
@router.delete("/profile/images/profile")
async def delete_profile_images(
    request: DeleteImagesRequest,
    current_user: tuple = Depends(get_current_user)
):
    """Delete specific profile images"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        current_images = profile_data.get("profile_images", [])
        
        # Remove images from storage
        for url in request.image_urls:
            if url in current_images:
                current_images.remove(url)
                delete_file_from_storage(
                    {"profile_images": [url]}, 
                    {"profile_images": "service_provider_images"}
                )
        
        profile_doc.update({
            "profile_images": current_images,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Images deleted successfully",
            "deleted_count": len(request.image_urls),
            "remaining_images": len(current_images)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profile/images/posters/{poster_id}")
async def delete_poster_image(
    poster_id: str,
    current_user: tuple = Depends(get_current_user)
):
    """Delete a specific poster by ID"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        current_posters = profile_data.get("poster_images", [])
        
        poster_to_delete = None
        updated_posters = []
        
        for poster in current_posters:
            if poster.get("id") == poster_id:
                poster_to_delete = poster
            else:
                updated_posters.append(poster)
        
        if not poster_to_delete:
            raise HTTPException(status_code=404, detail="Poster not found")
        
        # Delete from storage
        try:
            image_url = poster_to_delete.get("image_url")
            if image_url:
                delete_file_from_storage(
                    {"poster_images": [image_url]}, 
                    {"poster_images": "service_provider_posters"}
                )
        except Exception as storage_error:
            print(f"Warning: Failed to delete image from storage: {storage_error}")
        
        profile_doc.update({
            "poster_images": updated_posters,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Poster deleted successfully",
            "deleted_poster_id": poster_id,
            "remaining_posters": len(updated_posters)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile/toggle-status")
async def toggle_profile_status(current_user: tuple = Depends(get_current_user)):
    """Toggle service provider profile active/inactive status"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        current_status = profile_data.get("is_active", True)
        new_status = not current_status
        
        update_data = {
            "is_active": new_status,
            "updated_at": datetime.now().isoformat()
        }
        
        profile_doc.update(update_data)
        
        return {
            "message": f"Profile {'activated' if new_status else 'deactivated'} successfully",
            "is_active": new_status,
            "updated_at": update_data["updated_at"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/stats")
async def get_dashboard_stats(current_user: tuple = Depends(get_current_user)):
    """Get dashboard statistics for service provider"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid).get()
        
        if not profile_doc.exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile_doc.to_dict()
        
        completion_score = calculate_profile_completion(profile_data)
        
        stats = {
            "profile_completion": completion_score,
            "is_active": profile_data.get("is_active", True),
            "profile_images_count": len(profile_data.get("profile_images", [])),
            "poster_images_count": len(profile_data.get("poster_images", [])),
            "amenities_count": len(profile_data.get("amenities", [])),
            "has_coordinates": bool(profile_data.get("coordinates")),
            "has_operating_hours": bool(profile_data.get("operating_hours")),
            "has_social_media": bool(profile_data.get("social_media")),
        }
        
        return {"stats": stats}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_profile_completion(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate profile completion percentage"""
    
    required_fields = ["service_name", "description", "address", "phone_number", "service_category"]
    required_completed = sum(1 for field in required_fields if profile_data.get(field))
    required_percentage = (required_completed / len(required_fields)) * 60
    
    optional_score = 0
    if profile_data.get("profile_images"): optional_score += 15
    if profile_data.get("poster_images"): optional_score += 10
    if profile_data.get("operating_hours"): optional_score += 5
    if profile_data.get("coordinates"): optional_score += 5
    if profile_data.get("social_media"): optional_score += 3
    if profile_data.get("amenities"): optional_score += 2
    
    total_percentage = required_percentage + optional_score
    
    return {
        "total_percentage": int(min(total_percentage, 100)),
        "required_completed": required_completed,
        "required_total": len(required_fields),
        "has_profile_images": bool(profile_data.get("profile_images")),
        "has_poster_images": bool(profile_data.get("poster_images")),
        "has_operating_hours": bool(profile_data.get("operating_hours")),
        "has_coordinates": bool(profile_data.get("coordinates")),
        "has_social_media": bool(profile_data.get("social_media")),
        "has_amenities": bool(profile_data.get("amenities"))
    }