from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, status, Depends, UploadFile, File, Form, Query
from firebase_admin import auth
from app.database.connection import db
from app.models.user import BaseServiceProfile, UpdateAmenities, UpdateOperatingHours, UpdateProfileBasicInfo, UpdateSocialMedia 
from app.utils.storage_handle import upload_file_to_storage, delete_file_from_storage
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
import math

router = APIRouter()
profiles_collection = db.collection("service_provider_profiles")
users_collection = db.collection("users")
security = HTTPBearer()

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth in kilometers"""
    R = 6371  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi/2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current user"""
    try:
        token = credentials.credentials
        decoded_token = auth.verify_id_token(token)
        uid = decoded_token["uid"]
        
        # Check if user is a service provider
        user_doc = users_collection.document(uid).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data = user_doc.to_dict()
        if user_data.get("role") != "service_provider":
            raise HTTPException(status_code=403, detail="Access denied. Service provider role required")
        
        return uid, user_data
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication token")


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

        return {"message": "Current service provider fetched successfully", "data": response_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile/basic-info")
async def update_basic_info(
    service_name: str = Form(...),
    description: str = Form(...),
    address: str = Form(...),
    district: str = Form(...),
    phone_number: str = Form(...),
    email: Optional[str] = Form(None),
    website: Optional[str] = Form(None),
    latitude: Optional[float] = Form(None),
    longitude: Optional[float] = Form(None),
    current_user: tuple = Depends(get_current_user)
):
    """Update basic information of service provider profile"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        
        # Check if profile exists
        if not profile_doc.get().exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Prepare coordinates
        coordinates = None
        if latitude is not None and longitude is not None:
            coordinates = {"lat": latitude, "lng": longitude}
        
        # Update basic info
        update_data = {
            "service_name": service_name.strip(),
            "description": description.strip(),
            "address": address.strip(),
            "district": district.strip(),
            "phone_number": phone_number.strip(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Add optional fields if provided
        if email:
            update_data["email"] = email
        if website:
            update_data["website"] = website
        if coordinates:
            update_data["coordinates"] = coordinates
        
        profile_doc.update(update_data)
        
        return {
            "message": "Basic information updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/operating-hours")
async def update_operating_hours(
    request: UpdateOperatingHours,
    current_user: tuple = Depends(get_current_user)
):
    """Update operating hours"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        
        if not profile_doc.get().exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Convert Pydantic models to dict
        operating_hours_dict = {
            day: {"open": slot.open, "close": slot.close} 
            for day, slot in request.operating_hours.items()
        }
        
        update_data = {
            "operating_hours": operating_hours_dict,
            "updated_at": datetime.now().isoformat()
        }
        
        profile_doc.update(update_data)
        
        return {
            "message": "Operating hours updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile/social-media")
async def update_social_media(
    request: UpdateSocialMedia,  # Change this - use Pydantic model
    current_user: tuple = Depends(get_current_user)
):
    """Update social media links"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        
        if not profile_doc.get().exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        update_data = {
            "social_media": request.social_media,  # Access from model
            "updated_at": datetime.now().isoformat()
        }
        
        profile_doc.update(update_data)
        
        return {
            "message": "Social media links updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.put("/profile/amenities")
async def update_amenities(
    request: UpdateAmenities, 
    current_user: tuple = Depends(get_current_user)
):
    """Update amenities/features"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        
        if not profile_doc.get().exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        update_data = {
            "amenities": request.amenities,  # Access from model
            "updated_at": datetime.now().isoformat()
        }
        
        profile_doc.update(update_data)
        
        return {
            "message": "Amenities updated successfully",
            "updated_at": update_data["updated_at"]
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
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
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        for img in images:
            if img.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {img.content_type}")
        
        # Upload images
        image_urls = []
        for img in images:
            url = upload_file_to_storage(img, "service_provider_images")
            if url:
                image_urls.append(url)
        
        # Get current images and append new ones
        current_images = profile_data.get("profile_images", [])
        current_images.extend(image_urls)
        
        # Update profile
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


@router.post("/profile/images/posters")
async def upload_poster_images(
    posters: List[UploadFile] = File(...),
    current_user: tuple = Depends(get_current_user)
):
    """Upload poster images (promotional images)"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Validate file types
        allowed_types = ["image/jpeg", "image/jpg", "image/png"]
        for poster in posters:
            if poster.content_type not in allowed_types:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {poster.content_type}")
        
        # Upload posters
        poster_urls = []
        for poster in posters:
            url = upload_file_to_storage(poster, "service_provider_posters")
            if url:
                poster_urls.append(url)
        
        # Get current posters and append new ones
        current_posters = profile_data.get("poster_images", [])
        current_posters.extend(poster_urls)
        
        # Update profile
        profile_doc.update({
            "poster_images": current_posters,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": f"{len(poster_urls)} posters uploaded successfully",
            "uploaded_posters": poster_urls,
            "total_posters": len(current_posters)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profile/images/profile")
async def delete_profile_images(
    image_urls: List[str],
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
        for url in image_urls:
            if url in current_images:
                current_images.remove(url)
                delete_file_from_storage(
                    {"profile_images": [url]}, 
                    {"profile_images": "service_provider_images"}
                )
        
        # Update profile
        profile_doc.update({
            "profile_images": current_images,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Images deleted successfully",
            "deleted_count": len(image_urls),
            "remaining_images": len(current_images)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/profile/images/posters")
async def delete_poster_images(
    poster_urls: List[str],
    current_user: tuple = Depends(get_current_user)
):
    """Delete specific poster images"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid)
        profile_data = profile_doc.get().to_dict()
        
        if not profile_data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        current_posters = profile_data.get("poster_images", [])
        
        # Remove posters from storage
        for url in poster_urls:
            if url in current_posters:
                current_posters.remove(url)
                delete_file_from_storage(
                    {"poster_images": [url]}, 
                    {"poster_images": "service_provider_posters"}
                )
        
        # Update profile
        profile_doc.update({
            "poster_images": current_posters,
            "updated_at": datetime.now().isoformat()
        })
        
        return {
            "message": "Posters deleted successfully",
            "deleted_count": len(poster_urls),
            "remaining_posters": len(current_posters)
        }
    
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
            raise HTTPException(status_code=404, detail="Service provider is currently inactive")
        
        # Get basic user info
        user_doc = users_collection.document(provider_uid).get()
        if user_doc.exists:
            user_data = user_doc.to_dict()
            profile_data["provider_info"] = {
                "full_name": user_data.get("full_name"),
                "status": user_data.get("status")
            }
        
        profile_data["uid"] = provider_uid
        
        return {"profile": profile_data}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/all")
async def get_all_service_providers(
    service_category: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(50, le=100),
    offset: int = Query(0)
):
    """Get all service provider profiles with filters"""
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
            profile_data["uid"] = doc.id
            
            # Get user info
            user_doc = users_collection.document(doc.id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                profile_data["provider_name"] = user_data.get("full_name", "")
            
            profiles.append(profile_data)
        
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/search")
async def search_service_providers(
    service_category: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    active_only: bool = Query(True),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Search and filter service provider profiles (returns summary data)"""
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
            user_doc = users_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            # Create summary for search results
            profile_summary = {
                "uid": doc.id,
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "district": profile_data.get("district", ""),
                "description": profile_data.get("description", "")[:200],
                "profile_images": profile_data.get("profile_images", [])[:3],  # First 3 images
                "poster_images": profile_data.get("poster_images", [])[:2],    # First 2 posters
                "is_active": profile_data.get("is_active", True),
                "provider_name": user_data.get("full_name", ""),
                "coordinates": profile_data.get("coordinates"),
                "phone_number": profile_data.get("phone_number"),
                "amenities": profile_data.get("amenities", []),
                "created_at": profile_data.get("created_at")
            }
            
            profiles.append(profile_summary)
        
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/nearby")
async def get_nearby_service_providers(
    latitude: float = Query(..., description="User's current latitude"),
    longitude: float = Query(..., description="User's current longitude"),
    radius_km: float = Query(10, description="Search radius in kilometers"),
    service_category: Optional[str] = Query(None),
    limit: int = Query(20, le=100)
):
    """Get service providers within a specific radius"""
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
                distance = haversine(latitude, longitude, dest_lat, dest_lng)
                
                if distance <= radius_km:
                    # Get user info
                    user_doc = users_collection.document(doc.id).get()
                    user_data = user_doc.to_dict() if user_doc.exists else {}
                    
                    provider_summary = {
                        "uid": doc.id,
                        "service_name": profile_data.get("service_name", ""),
                        "service_category": profile_data.get("service_category", ""),
                        "district": profile_data.get("district", ""),
                        "description": profile_data.get("description", "")[:200],
                        "profile_images": profile_data.get("profile_images", [])[:3],
                        "poster_images": profile_data.get("poster_images", [])[:2],
                        "provider_name": user_data.get("full_name", ""),
                        "coordinates": coordinates,
                        "phone_number": profile_data.get("phone_number"),
                        "distance_km": round(distance, 2),
                        "amenities": profile_data.get("amenities", [])
                    }
                    
                    nearby_providers.append(provider_summary)
        
        # Sort by distance
        nearby_providers.sort(key=lambda x: x["distance_km"])
        
        # Apply limit
        nearby_providers = nearby_providers[:limit]
        
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
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/profiles/by-category/{service_category}")
async def get_providers_by_category(
    service_category: str,
    district: Optional[str] = Query(None),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all providers in a specific category"""
    try:
        query = profiles_collection.where("service_category", "==", service_category).where("is_active", "==", True)
        
        if district:
            query = query.where("district", "==", district)
        
        docs = query.limit(limit).offset(offset).stream()
        
        providers = []
        for doc in docs:
            profile_data = doc.to_dict()
            
            user_doc = users_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            provider_summary = {
                "uid": doc.id,
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "district": profile_data.get("district", ""),
                "description": profile_data.get("description", "")[:200],
                "profile_images": profile_data.get("profile_images", [])[:3],
                "poster_images": profile_data.get("poster_images", [])[:2],
                "provider_name": user_data.get("full_name", ""),
                "coordinates": profile_data.get("coordinates"),
                "phone_number": profile_data.get("phone_number"),
                "amenities": profile_data.get("amenities", [])
            }
            
            providers.append(provider_summary)
        
        return {
            "service_category": service_category,
            "count": len(providers),
            "profiles": providers,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(providers) == limit
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            
            user_doc = users_collection.document(doc.id).get()
            user_data = user_doc.to_dict() if user_doc.exists else {}
            
            provider_summary = {
                "uid": doc.id,
                "service_name": profile_data.get("service_name", ""),
                "service_category": profile_data.get("service_category", ""),
                "district": profile_data.get("district", ""),
                "description": profile_data.get("description", "")[:200],
                "profile_images": profile_data.get("profile_images", [])[:3],
                "poster_images": profile_data.get("poster_images", [])[:2],
                "provider_name": user_data.get("full_name", ""),
                "coordinates": profile_data.get("coordinates"),
                "phone_number": profile_data.get("phone_number"),
                "amenities": profile_data.get("amenities", [])
            }
            
            providers.append(provider_summary)
        
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
        
        return {"stats": stats}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def calculate_profile_completion(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate profile completion percentage"""
    
    # Required fields (60% weight)
    required_fields = ["service_name", "description", "address", "phone_number", "service_category"]
    required_completed = sum(1 for field in required_fields if profile_data.get(field))
    required_percentage = (required_completed / len(required_fields)) * 60
    
    # Optional fields (40% weight)
    optional_score = 0
    
    if profile_data.get("profile_images"):
        optional_score += 15  # Images are important
    
    if profile_data.get("poster_images"):
        optional_score += 10  # Posters add value
    
    if profile_data.get("operating_hours"):
        optional_score += 5
    
    if profile_data.get("coordinates"):
        optional_score += 5
    
    if profile_data.get("social_media"):
        optional_score += 3
    
    if profile_data.get("amenities"):
        optional_score += 2
    
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