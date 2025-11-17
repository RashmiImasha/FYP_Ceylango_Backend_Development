import uuid, logging
from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, Query
from firebase_admin import auth
from app.models.user import ( UpdateProfileRequest,)
from app.utils.crud_utils import CrudUtils
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from pydantic import BaseModel
from app.database.connection import profiles_collection, user_collection

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)

class PosterMetadata(BaseModel):
    name: str
    description: str
    expiration_date: Optional[str] = None

class UpdatePosterMetadataRequest(BaseModel):
    poster_id: str
    name: str
    description: str
    expiration_date: Optional[str] = None

class DeleteImagesRequest(BaseModel):
    image_urls: List[str]


# ===== AUTH DEPENDENCY =====
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        
        user_doc = user_collection.document(uid).get()
        if not user_doc.exists:
            logger.warning(f"User with UID {uid} not found in Firestore")
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        if user_data.get("role") != "service_provider":
            logger.warning(f"Access denied for UID {uid}: not a service provider")
            raise HTTPException(status_code=403, detail="Access denied. Service provider role required")
               
        return uid, user_data
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
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

        logger.info(f"Profile fetched for UID {uid}")        
        return {
            "profile": profile_data,
            "user_info": {
                "email": user_data.get("email"),
                "full_name": user_data.get("full_name"),
                "status": user_data.get("status")
            }
        }
            
    except Exception as e:
        logger.error(f"Error fetching profile for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/me")
async def get_current_service_provider(
    current_user: tuple = Depends(get_current_user)
):
    """
    Get currently authenticated service provider's profile + user info.
    """
    uid, user_data = current_user

    try:
        # Fetch service provider profile
        profile_doc = profiles_collection.document(uid).get()
        if not profile_doc.exists:            
            raise HTTPException(status_code=404, detail="Service provider profile not found")

        profile_data = profile_doc.to_dict()

        # Merge both Firestore user info and service provider profile
        response_data = {
            "uid": uid,
            "email": user_data.get("email"),
            "full_name": user_data.get("full_name"),
            "role": user_data.get("role"),
            "status": user_data.get("status", "active"),
            "profile": {
                "service_name": profile_data.get("service_name"),
                "service_category": profile_data.get("service_category"),
                "description": profile_data.get("description"),
                "address": profile_data.get("address"),
                "district": profile_data.get("district"),
                "phone_number": profile_data.get("phone_number"),
                "email": profile_data.get("email"),
                "website": profile_data.get("website"),
                "coordinates": profile_data.get("coordinates"),
                "amenities": profile_data.get("amenities", []),
                "operating_hours": profile_data.get("operating_hours"),
                "social_media": profile_data.get("social_media"),
                "profile_images": profile_data.get("profile_images", []),
                "poster_images": profile_data.get("poster_images", []),
                "is_active": profile_data.get("is_active", True),
                "created_at": profile_data.get("created_at"),
                "updated_at": profile_data.get("updated_at")
            }
        }

        logger.info(f"Current service provider data fetched for UID {uid}")
        return {"message": "Current service provider fetched successfully", "data": response_data}

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error fetching current service provider for UID {uid}: {str(e)}")
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

        logger.info(f"Profile updated for UID {uid}")        
        return {
            "message": "Profile updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FormData only for file uploads
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
                logger.error(f"Unsupported file type attempted for upload by UID {uid}: {img.content_type}")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file type: {img.content_type}"
                )
        
        # Upload images
        image_urls = []
        for img in images:
            url = CrudUtils.upload_file_to_storage(img, "service_provider_images")
            if url:
                image_urls.append(url)
        
        # Get current images and append new ones
        current_images = profile_data.get("profile_images", [])
        current_images.extend(image_urls)
        
        profile_doc.update({
            "profile_images": current_images,
            "updated_at": datetime.now().isoformat()
        })

        logger.info(f"{len(image_urls)} profile images uploaded for UID {uid}")        
        return {
            "message": f"{len(image_urls)} images uploaded successfully",
            "uploaded_images": image_urls,
            "total_images": len(current_images)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading profile images for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# FormData for file + metadata as JSON string
@router.post("/profile/images/posters")
async def upload_poster_image(
    poster: UploadFile = File(...),
    metadata: str = Form(...),  # JSON string
    current_user: tuple = Depends(get_current_user)
):
    """Upload a promotional poster with metadata (expiration_date is optional)"""
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
            logger.error(f"Unsupported poster file type upload by UID {uid}: {poster.content_type}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {poster.content_type}"
            )
        
        # Upload poster image
        image_url = CrudUtils.upload_file_to_storage(poster, "service_provider_posters")
        
        if not image_url:
            raise HTTPException(status_code=500, detail="Failed to upload image")
        
        # Create poster object
        poster_id = str(uuid.uuid4())
        poster_data = {
            "id": poster_id,
            "name": poster_metadata.name.strip(),
            "description": poster_metadata.description.strip(),
            "image_url": image_url,
            "created_at": datetime.now().isoformat()
        }
        
        # Only add expiration_date if provided
        if poster_metadata.expiration_date:
            poster_data["expiration_date"] = poster_metadata.expiration_date
        
        current_posters = profile_data.get("poster_images", [])
        current_posters.append(poster_data)
        
        profile_doc.update({
            "poster_images": current_posters,
            "updated_at": datetime.now().isoformat()
        })

        logger.info(f"Poster uploaded for UID {uid} with ID {poster_id}")        
        return {
            "message": "Poster uploaded successfully",
            "poster": poster_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading poster for UID {uid}: {str(e)}")
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

        logger.info(f"Poster metadata updated for UID {uid}, Poster ID {request.poster_id}")        
        return {
            "message": "Poster metadata updated successfully",
            "poster_id": request.poster_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating poster metadata: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
# JSON endpoint for deletions
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
                CrudUtils.delete_file_from_storage(
                    {"profile_images": [url]}, 
                    {"profile_images": "service_provider_images"}
                )
        
        profile_doc.update({
            "profile_images": current_images,
            "updated_at": datetime.now().isoformat()
        })
        logger.info(f"{len(request.image_urls)} profile images deleted for UID {uid}")
        return {
            "message": "Images deleted successfully",
            "deleted_count": len(request.image_urls),
            "remaining_images": len(current_images)
        }
    
    except Exception as e:
        logger.error(f"Error deleting profile images for UID {uid}: {str(e)}")
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
                CrudUtils.delete_file_from_storage(
                    {"poster_images": [image_url]}, 
                    {"poster_images": "service_provider_posters"}
                )
        except Exception as storage_error:
            logger.error(f"Failed to delete image from storage: {storage_error}")
        
        profile_doc.update({
            "poster_images": updated_posters,
            "updated_at": datetime.now().isoformat()
        })
        logger.info(f"Poster deleted successful for UID {uid}, Poster ID {poster_id}")
        return {
            "message": "Poster deleted successfully",
            "deleted_poster_id": poster_id,
            "remaining_posters": len(updated_posters)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting poster for UID {uid}, Poster ID {poster_id}: {str(e)}")
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
        logger.info(f"Profile status toggled for UID {uid} to {'active' if new_status else 'inactive'}")        
        return {
            "message": f"Profile {'activated' if new_status else 'deactivated'} successfully",
            "is_active": new_status,
            "updated_at": update_data["updated_at"]
        }
    
    except Exception as e:
        logger.error(f"Error toggling profile status for UID {uid}: {str(e)}")
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
        logger.info(f"Dashboard stats fetched for UID {uid}")
        return {"stats": stats}
    
    except Exception as e:
        logger.error(f"Error fetching dashboard stats for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{provider_uid}")
async def get_service_provider_profile(provider_uid: str):

    """Get public profile of any service provider (for customers)"""
    try:
        profile_doc = profiles_collection.document(provider_uid).get()
        
        if not profile_doc.exists:
            raise HTTPException(status_code=404, detail="Service provider profile not found")
        
        profile_data = profile_doc.to_dict()
        
        # Only return active profiles to public
        if not profile_data.get("is_active", True):
            logger.error(f"Attempt to access inactive profile for UID {provider_uid}")
            raise HTTPException(status_code=404, detail="Service provider is currently inactive")
        
        # Get basic user info
        user_doc = user_collection.document(provider_uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            profile_data["provider_info"] = {
                "full_name": user_data.get("full_name"),
                "status": user_data.get("status")
            }
        
        profile_data["uid"] = provider_uid

        logger.info(f"Public profile fetched for UID {provider_uid}")        
        return {"profile": profile_data}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching public profile for UID {provider_uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profiles/all")
async def get_all_service_providers():
    try:        
        return CrudUtils.get_all(profiles_collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/profiles/search")
async def search_service_providers(
    service_category: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Search and filter service provider profiles (returns complete data)"""
    try:
        query = profiles_collection
        
        if service_category:
            query = query.where("service_category", "==", service_category)
        
        if district:
            query = query.where("district", "==", district)
        
        if active_only:
            query = query.where("is_active", "==", True)
        
        # Apply pagination
        docs = query.limit(limit).offset(offset).stream()
        
        profiles = []
        for doc in docs:
            profile_data = doc.to_dict()
            
            # Get user info
            user_doc = user_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            # Complete profile data
            profile_complete = {
                "uid": doc.id,
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "description": profile_data.get("description", ""),
                "address": profile_data.get("address", ""),
                "district": profile_data.get("district", ""),
                "coordinates": profile_data.get("coordinates"),
                "phone_number": profile_data.get("phone_number", ""),
                "email": profile_data.get("email"),
                "website": profile_data.get("website"),
                "social_media": profile_data.get("social_media", {}),
                "operating_hours": profile_data.get("operating_hours", {}),
                "profile_images": profile_data.get("profile_images", []),
                "poster_images": profile_data.get("poster_images", []),
                "amenities": profile_data.get("amenities", []),
                "is_active": profile_data.get("is_active", True),
                "created_at": profile_data.get("created_at"),
                "updated_at": profile_data.get("updated_at"),
                "provider_info": {
                    "full_name": user_data.get("full_name", ""),
                    "status": user_data.get("status", "")
                }
            }
            
            profiles.append(profile_complete)
        
        logger.info(f"Service provider profiles searched with filters - Category: {service_category}, District: {district}, Active Only: {active_only}")
        
        return {
            "count": len(profiles),
            "profiles": profiles,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(profiles) == limit
            },
            "filters": {
                "service_category": service_category,
                "district": district,
                "active_only": active_only
            }
        }
    except Exception as e:
        logger.error(f"Error searching service provider profiles: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/profiles/nearby")
async def get_nearby_service_providers(
    latitude: float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_km: float = Query(10, description="Search radius in kilometers"),
    service_category: Optional[str] = Query(None),
    limit: int = Query(20, le=100)
):
    """Get service providers within a specific radius with complete data"""
    try:
        query = profiles_collection.where("is_active", "==", True)
        
        if service_category:
            query = query.where("service_category", "==", service_category)
        
        docs = query.stream()
        
        nearby_providers = []
        
        for doc in docs:
            profile_data = doc.to_dict()
            coordinates = profile_data.get("coordinates")
            
            if coordinates and "lat" in coordinates and "lng" in coordinates:
                dest_lat = coordinates["lat"]
                dest_lng = coordinates["lng"]
                
                # Calculate distance
                distance = CrudUtils.haversine(latitude, longitude, dest_lat, dest_lng)
                
                if distance <= radius_km:
                    # Get user info
                    user_doc = user_collection.document(doc.id).get()
                    user_data = user_doc.to_dict() if user_doc.exists else {}
                    
                    # Complete profile with distance
                    provider_complete = {
                        "uid": doc.id,
                        "service_name": profile_data.get("service_name", ""),
                        "service_category": profile_data.get("service_category", ""),
                        "description": profile_data.get("description", ""),
                        "address": profile_data.get("address", ""),
                        "district": profile_data.get("district", ""),
                        "coordinates": coordinates,
                        "phone_number": profile_data.get("phone_number", ""),
                        "email": profile_data.get("email"),
                        "website": profile_data.get("website"),
                        "social_media": profile_data.get("social_media", {}),
                        "operating_hours": profile_data.get("operating_hours", {}),
                        "profile_images": profile_data.get("profile_images", []),
                        "poster_images": profile_data.get("poster_images", []),
                        "amenities": profile_data.get("amenities", []),
                        "is_active": profile_data.get("is_active", True),
                        "created_at": profile_data.get("created_at"),
                        "updated_at": profile_data.get("updated_at"),
                        "provider_info": {
                            "full_name": user_data.get("full_name", ""),
                            "status": user_data.get("status", "")
                        },
                        "distance_km": round(distance, 2)
                    }
                    
                    nearby_providers.append(provider_complete)
        
        # Sort by distance
        nearby_providers.sort(key=lambda x: x["distance_km"])
        
        # Apply limit
        nearby_providers = nearby_providers[:limit]

        logger.info(f"Found {len(nearby_providers)} nearby service providers within {radius_km} km")
        
        if not nearby_providers:
            return {
                "count": 0,
                "profiles": [],
                "message": f"No service providers found within {radius_km} km"
            }
        
        return {
            "count": len(nearby_providers),
            "profiles": nearby_providers,
            "search_params": {
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
                "service_category": service_category
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching nearby service providers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/profiles/by-category/{service_category}")
async def get_providers_by_category(
    service_category: str,
    district: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all providers in a specific category with complete profile data"""
    try:
        # Validate service_category
        if not service_category or not service_category.strip():
            raise HTTPException(status_code=400, detail="Service category is required")
        
        # Build query
        query = profiles_collection.where("service_category", "==", service_category.strip()).where("is_active", "==", True)
        
        if district and district.strip():
            query = query.where("district", "==", district.strip())
        
        # Apply pagination
        docs = query.limit(limit).offset(offset).stream()
        
        providers = []
        for doc in docs:
            profile_data = doc.to_dict()
            
            # Get user info
            user_doc = user_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            # Build complete provider profile
            provider_profile = {
                "uid": doc.id,
                # Basic service information
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "description": profile_data.get("description", ""),
                "address": profile_data.get("address", ""),
                "district": profile_data.get("district", ""),
                
                # Contact information
                "phone_number": profile_data.get("phone_number", ""),
                "email": profile_data.get("email"),
                "website": profile_data.get("website"),
                
                # Location
                "coordinates": profile_data.get("coordinates"),
                
                # Operating information
                "operating_hours": profile_data.get("operating_hours", {}),
                "social_media": profile_data.get("social_media", {}),
                
                # Media
                "profile_images": profile_data.get("profile_images", []),  # Regular images
                "poster_images": profile_data.get("poster_images", []),    # Promotional posters
                
                # Features
                "amenities": profile_data.get("amenities", []),
                
                # Status and metadata
                "is_active": profile_data.get("is_active", True),
                "created_at": profile_data.get("created_at"),
                "updated_at": profile_data.get("updated_at"),
                
                # Provider personal info
                "provider_info": {
                    "full_name": user_data.get("full_name", ""),
                    "email": user_data.get("email", ""),
                    "status": user_data.get("status", "")
                }
            }
            
            providers.append(provider_profile)
        
        logger.info(f"Fetched {len(providers)} providers in category {service_category} with district filter: {district}")
        return {
            "service_category": service_category,
            "count": len(providers),
            "profiles": providers,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(providers) == limit
            },
            "filters_applied": {
                "district": district if district else "All districts"
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching providers by category {service_category}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error while fetching service providers")
    
@router.get("/profiles/by-district/{district}")
async def get_providers_by_district(
    district: str,
    service_category: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all providers in a specific district"""
    try:
        query = profiles_collection.where("district", "==", district).where("is_active", "==", True)
        
        if service_category:
            query = query.where("service_category", "==", service_category)
        
        docs = query.limit(limit).offset(offset).stream()
        
        providers = []
        for doc in docs:
            profile_data = doc.to_dict()
            
            user_doc = user_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            provider_summary = {
                "uid": doc.id,
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "description": profile_data.get("description", ""),
                "address": profile_data.get("address", ""),
                "district": profile_data.get("district", ""),
                "coordinates": profile_data.get("coordinates"),
                "phone_number": profile_data.get("phone_number", ""),
                "email": profile_data.get("email"),
                "website": profile_data.get("website"),
                "social_media": profile_data.get("social_media", {}),
                "operating_hours": profile_data.get("operating_hours", {}),
                "profile_images": profile_data.get("profile_images", []),
                "poster_images": profile_data.get("poster_images", []),
                "amenities": profile_data.get("amenities", []),
                "is_active": profile_data.get("is_active", True),
                "created_at": profile_data.get("created_at"),
                "updated_at": profile_data.get("updated_at"),
                "provider_info": {
                    "full_name": user_data.get("full_name", ""),
                    "status": user_data.get("status", "")
            }}
            
            providers.append(provider_summary)
        
        logger.info(f"Fetched {len(providers)} providers in district {district} with category filter: {service_category}")
        return {
            "district": district,
            "count": len(providers),
            "profiles": providers,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(providers) == limit
            }
        }    
    except Exception as e:
        logger.error(f"Error fetching providers by district {district}: {str(e)}")
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
        
        # Calculate completion percentage
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
            "profile_views": 0,  # TODO: Implement view tracking
            "total_bookings": 0,  # TODO: Implement booking system
        }     
        logger.info(f"Dashboard stats fetched for UID {uid}")   
        return {"stats": stats}
    
    except Exception as e:
        logger.error(f"Error fetching dashboard stats for UID {uid}: {str(e)}")
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

    logger.info(f"Profile completion calculated: {int(min(total_percentage, 100))}%")    
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