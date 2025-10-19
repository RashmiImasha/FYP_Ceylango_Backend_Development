# from typing import Optional, Dict, Any, Union
# from fastapi import APIRouter, HTTPException, status, Depends
# from firebase_admin import auth
# from app.database.connection import db
# from app.models.user import (
#     ServiceProviderProfile, BaseServiceProfile, MainCategory,
#     AccommodationProfile, FoodDiningProfile, WellnessProfile,
#     ShoppingProfile, ActivitiesProfile, TransportationProfile
# )
# from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
# from datetime import datetime
# from app.utils.validators import ProfileValidator

# router = APIRouter()
# profiles_collection = db.collection("service_provider_profiles")
# users_collection = db.collection("users")
# security = HTTPBearer()

# async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
#     """Verify JWT token and return current user"""
#     try:
#         token = credentials.credentials
#         decoded_token = auth.verify_id_token(token)
#         uid = decoded_token["uid"]
        
#         # Check if user is a service provider
#         user_doc = users_collection.document(uid).get()
#         if not user_doc.exists:
#             raise HTTPException(status_code=404, detail="User not found")
        
#         user_data = user_doc.to_dict()
#         if user_data.get("role") != "service_provider":
#             raise HTTPException(status_code=403, detail="Access denied. Service provider role required")
        
#         return uid, user_data
#     except Exception as e:
#         raise HTTPException(status_code=401, detail="Invalid authentication token")

# @router.get("/profile")
# async def get_my_profile(current_user: tuple = Depends(get_current_user)):
#     """Get service provider's own profile"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid).get()
        
#         if not profile_doc.exists:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         profile_data = profile_doc.to_dict()
        
#         return {
#             "profile": profile_data,
#             "user_info": {
#                 "email": user_data.get("email"),
#                 "full_name": user_data.get("full_name"),
#                 "status": user_data.get("status")
#             }
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.put("/profile/base-info")
# async def update_base_info(
#     base_info: BaseServiceProfile,
#     current_user: tuple = Depends(get_current_user)
# ):
#     """Update base information of service provider profile"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid)
        
#         # Check if profile exists
#         if not profile_doc.get().exists:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         # Validate base info
#         base_info_dict = base_info.dict()
#         validation_errors = ProfileValidator.validate_base_info(base_info_dict)
        
#         if validation_errors:
#             raise HTTPException(status_code=422, detail={
#                 "message": "Validation failed",
#                 "errors": validation_errors
#             })
        
#         # Update base info
#         update_data = {
#             "base_info": base_info_dict,
#             "updated_at": datetime.now().isoformat()
#         }
        
#         profile_doc.update(update_data)
        
#         return {
#             "message": "Base information updated successfully",
#             "updated_at": update_data["updated_at"]
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.put("/profile/category-data")
# async def update_category_data(
#     category_data: Union[
#         AccommodationProfile,
#         FoodDiningProfile,
#         WellnessProfile,
#         ShoppingProfile,
#         ActivitiesProfile,
#         TransportationProfile,
#         Dict[str, Any]
#     ],
#     current_user: tuple = Depends(get_current_user)
# ):
#     """Update category-specific data of service provider profile"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid)
#         profile_data = profile_doc.get().to_dict()
        
#         if not profile_data:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         # Get the service provider's category
#         main_category = profile_data.get("main_category")
        
#         # Convert Pydantic model to dict if needed
#         if hasattr(category_data, 'dict'):
#             category_data_dict = category_data.dict()
#         else:
#             category_data_dict = category_data
        
#         # Validate category data
#         validation_errors = ProfileValidator.validate_category_data(main_category, category_data_dict)
        
#         if validation_errors:
#             raise HTTPException(status_code=422, detail={
#                 "message": "Validation failed",
#                 "errors": validation_errors
#             })
        
#         # Update category data
#         update_data = {
#             "category_data": category_data_dict,
#             "updated_at": datetime.now().isoformat(),
#             "profile_completed": True  # Mark as completed when category data is updated
#         }
        
#         profile_doc.update(update_data)
        
#         return {
#             "message": "Category-specific data updated successfully",
#             "main_category": main_category,
#             "updated_at": update_data["updated_at"]
#         }
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/profile/template")
# async def get_profile_template(current_user: tuple = Depends(get_current_user)):
#     """Get empty profile template based on service provider's category"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid).get()
        
#         if not profile_doc.exists:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         profile_data = profile_doc.to_dict()
#         main_category = profile_data.get("main_category")
#         sub_category = profile_data.get("sub_category")
        
#         # Return template based on category
#         template = get_category_template(main_category)
        
#         return {
#             "main_category": main_category,
#             "sub_category": sub_category,
#             "template": template,
#             "current_data": profile_data.get("category_data", {})
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def get_category_template(main_category: str) -> Dict[str, Any]:
#     """Return template structure for each category"""
#     templates = {
#         MainCategory.ACCOMMODATION: {
#             "room_types": [
#                 {
#                     "name": "Standard Room",
#                     "description": "Basic room with essential amenities",
#                     "price_per_night": 0,
#                     "max_occupancy": 2,
#                     "amenities": [],
#                     "images": []
#                 }
#             ],
#             "check_in_time": "14:00",
#             "check_out_time": "11:00",
#             "cancellation_policy": "Free cancellation up to 24 hours before check-in",
#             "room_amenities": ["Wi-Fi", "Air Conditioning", "Private Bathroom"],
#             "hotel_amenities": ["Reception", "Parking", "Restaurant"],
#             "price_range": {"min": 0, "max": 0}
#         },
        
#         MainCategory.FOOD_DINING: {
#             "cuisine_types": ["Local", "International"],
#             "menu_items": [
#                 {
#                     "category": "Main Course",
#                     "items": [
#                         {
#                             "name": "Rice and Curry",
#                             "description": "Traditional Sri Lankan meal",
#                             "price": 0,
#                             "dietary_options": ["vegetarian_available"],
#                             "spice_level": "medium"
#                         }
#                     ]
#                 }
#             ],
#             "dietary_options": ["vegetarian", "vegan", "gluten_free"],
#             "average_meal_price": {"min": 0, "max": 0},
#             "seating_capacity": 0,
#             "delivery_available": False,
#             "takeaway_available": True
#         },
        
#         MainCategory.WELLNESS: {
#             "services_offered": [
#                 {
#                     "service_name": "Ayurvedic Massage",
#                     "description": "Traditional healing massage",
#                     "duration_minutes": 60,
#                     "price": 0,
#                     "benefits": []
#                 }
#             ],
#             "therapists": [
#                 {
#                     "name": "Dr. Silva",
#                     "qualifications": "Certified Ayurvedic Practitioner",
#                     "experience_years": 5,
#                     "specializations": []
#                 }
#             ],
#             "treatment_packages": [],
#             "booking_advance_days": 7,
#             "cancellation_hours": 24
#         },
        
#         MainCategory.SHOPPING: {
#             "product_categories": ["Handicrafts", "Textiles", "Spices"],
#             "inventory_items": [
#                 {
#                     "product_name": "Handmade Mask",
#                     "category": "Handicrafts",
#                     "description": "Traditional Sri Lankan wooden mask",
#                     "price": 0,
#                     "stock_quantity": 0,
#                     "images": []
#                 }
#             ],
#             "payment_methods": ["Cash", "Card", "Mobile Payment"],
#             "shipping_available": False,
#             "return_policy": "7-day return policy for unused items"
#         },
        
#         MainCategory.ACTIVITIES: {
#             "activity_types": ["Cultural", "Adventure", "Nature"],
#             "duration": "2 hours",
#             "group_size": {"min": 1, "max": 10},
#             "difficulty_level": "Easy",
#             "equipment_provided": [],
#             "age_restrictions": "Suitable for all ages",
#             "price_per_person": 0,
#             "includes": [],
#             "excludes": []
#         },
        
#         MainCategory.TRANSPORTATION: {
#             "vehicle_types": [
#                 {
#                     "type": "Car",
#                     "model": "Toyota Prius",
#                     "capacity": 4,
#                     "price_per_day": 0,
#                     "price_per_km": 0,
#                     "features": ["AC", "GPS"]
#                 }
#             ],
#             "coverage_areas": [],
#             "booking_advance_hours": 2,
#             "driver_available": True,
#             "fuel_policy": "Full to Full",
#             "insurance_included": True,
#             "additional_services": ["Airport Pickup", "Tour Guide"]
#         }
#     }
    
#     return templates.get(main_category, {})

# @router.get("/profile/{provider_uid}")
# async def get_service_provider_profile(provider_uid: str):
#     """Get public profile of any service provider (for customers)"""
#     try:
#         profile_doc = profiles_collection.document(provider_uid).get()
        
#         if not profile_doc.exists:
#             raise HTTPException(status_code=404, detail="Service provider profile not found")
        
#         profile_data = profile_doc.to_dict()
        
#         # Only return public information
#         public_profile = {
#             "uid": profile_data.get("uid"),
#             "main_category": profile_data.get("main_category"),
#             "sub_category": profile_data.get("sub_category"),
#             "base_info": profile_data.get("base_info", {}),
#             "category_data": profile_data.get("category_data", {}),
#             "profile_completed": profile_data.get("profile_completed", False),
#             "created_at": profile_data.get("created_at")
#         }
        
#         # Get basic user info
#         user_doc = users_collection.document(provider_uid).get()
#         if user_doc.exists:
#             user_data = user_doc.to_dict()
#             public_profile["provider_info"] = {
#                 "full_name": user_data.get("full_name"),
#                 "email": user_data.get("email"),
#                 "status": user_data.get("status")
#             }
        
#         return {"profile": public_profile}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.put("/profile/toggle-status")
# async def toggle_profile_status(current_user: tuple = Depends(get_current_user)):
#     """Toggle service provider profile active/inactive status"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid)
#         profile_data = profile_doc.get().to_dict()
        
#         if not profile_data:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         current_status = profile_data.get("base_info", {}).get("is_active", True)
#         new_status = not current_status
        
#         update_data = {
#             "base_info.is_active": new_status,
#             "updated_at": datetime.now().isoformat()
#         }
        
#         profile_doc.update(update_data)
        
#         return {
#             "message": f"Profile {'activated' if new_status else 'deactivated'} successfully",
#             "is_active": new_status,
#             "updated_at": update_data["updated_at"]
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Additional utility routes

# @router.get("/profiles/search")
# async def search_service_providers(
#     main_category: Optional[MainCategory] = None,
#     sub_category: Optional[str] = None,
#     district: Optional[str] = None,
#     active_only: bool = True,
#     limit: int = 20,
#     offset: int = 0
# ):
#     """Search and filter service provider profiles for customers"""
#     try:
#         query = profiles_collection
        
#         if main_category:
#             query = query.where("main_category", "==", main_category)
        
#         if sub_category:
#             query = query.where("sub_category", "==", sub_category)
        
#         if district:
#             query = query.where("base_info.district", "==", district)
        
#         if active_only:
#             query = query.where("base_info.is_active", "==", True)
        
#         # Apply pagination
#         docs = query.limit(limit).offset(offset).stream()
        
#         profiles = []
#         for doc in docs:
#             profile_data = doc.to_dict()
            
#             # Get user info
#             user_doc = users_collection.document(doc.id).get()
#             user_data = user_doc.to_dict() if user_doc.exists else {}
            
#             # Create summary for search results
#             profile_summary = {
#                 "uid": doc.id,
#                 "service_name": profile_data.get("base_info", {}).get("service_name", ""),
#                 "main_category": profile_data.get("main_category"),
#                 "sub_category": profile_data.get("sub_category"),
#                 "district": profile_data.get("base_info", {}).get("district", ""),
#                 "description": profile_data.get("base_info", {}).get("description", "")[:200] + "...",
#                 "images": profile_data.get("base_info", {}).get("images", [])[:3],  # First 3 images
#                 "is_active": profile_data.get("base_info", {}).get("is_active", True),
#                 "profile_completed": profile_data.get("profile_completed", False),
#                 "provider_name": user_data.get("full_name", ""),
#                 "created_at": profile_data.get("created_at")
#             }
            
#             # Add category-specific preview data
#             category_data = profile_data.get("category_data", {})
#             if main_category == MainCategory.ACCOMMODATION:
#                 profile_summary["price_range"] = category_data.get("price_range", {})
#                 profile_summary["room_count"] = len(category_data.get("room_types", []))
                
#             elif main_category == MainCategory.FOOD_DINING:
#                 profile_summary["cuisine_types"] = category_data.get("cuisine_types", [])
#                 profile_summary["price_range"] = category_data.get("average_meal_price", {})
                
#             elif main_category == MainCategory.WELLNESS:
#                 profile_summary["service_count"] = len(category_data.get("services_offered", []))
                
#             elif main_category == MainCategory.ACTIVITIES:
#                 profile_summary["price_per_person"] = category_data.get("price_per_person")
#                 profile_summary["duration"] = category_data.get("duration")
            
#             profiles.append(profile_summary)
        
#         return {
#             "count": len(profiles),
#             "profiles": profiles,
#             "pagination": {
#                 "limit": limit,
#                 "offset": offset,
#                 "has_more": len(profiles) == limit
#             },
#             "filters": {
#                 "main_category": main_category,
#                 "sub_category": sub_category,
#                 "district": district,
#                 "active_only": active_only
#             }
#         }
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @router.get("/dashboard/stats")
# async def get_dashboard_stats(current_user: tuple = Depends(get_current_user)):
#     """Get dashboard statistics for service provider"""
#     uid, user_data = current_user
    
#     try:
#         profile_doc = profiles_collection.document(uid).get()
        
#         if not profile_doc.exists:
#             raise HTTPException(status_code=404, detail="Profile not found")
        
#         profile_data = profile_doc.to_dict()
#         base_info = profile_data.get("base_info", {})
#         category_data = profile_data.get("category_data", {})
        
#         # Calculate completion percentage
#         completion_score = calculate_profile_completion(profile_data)
        
#         stats = {
#             "profile_completion": completion_score,
#             "is_active": base_info.get("is_active", True),
#             "profile_views": 0,  # TODO: Implement view tracking
#             "total_bookings": 0,  # TODO: Implement booking system
#             "category_specific_stats": get_category_specific_stats(profile_data)
#         }
        
#         return {"stats": stats}
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# def calculate_profile_completion(profile_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Calculate profile completion percentage"""
#     base_info = profile_data.get("base_info", {})
#     category_data = profile_data.get("category_data", {})
    
#     # Base info completion (60% weight)
#     base_fields = ["service_name", "description", "address", "phone_number"]
#     base_completed = sum(1 for field in base_fields if base_info.get(field))
#     base_percentage = (base_completed / len(base_fields)) * 60
    
#     # Category data completion (30% weight)
#     category_completed = 30 if category_data else 0
    
#     # Additional info completion (10% weight)
#     additional_completed = 0
#     if base_info.get("images"):
#         additional_completed += 5
#     if base_info.get("operating_hours"):
#         additional_completed += 5
    
#     total_percentage = base_percentage + category_completed + additional_completed
    
#     return {
#         "total_percentage": int(total_percentage),
#         "base_info_completed": base_completed,
#         "base_info_total": len(base_fields),
#         "has_category_data": bool(category_data),
#         "has_images": bool(base_info.get("images")),
#         "has_operating_hours": bool(base_info.get("operating_hours"))
#     }

# def get_category_specific_stats(profile_data: Dict[str, Any]) -> Dict[str, Any]:
#     """Get category-specific statistics"""
#     main_category = profile_data.get("main_category")
#     category_data = profile_data.get("category_data", {})
    
#     if main_category == MainCategory.ACCOMMODATION:
#         return {
#             "total_rooms": len(category_data.get("room_types", [])),
#             "amenities_count": len(category_data.get("hotel_amenities", []))
#         }
#     elif main_category == MainCategory.FOOD_DINING:
#         return {
#             "menu_items": sum(len(cat.get("items", [])) for cat in category_data.get("menu_items", [])),
#             "cuisine_types": len(category_data.get("cuisine_types", []))
#         }
#     elif main_category == MainCategory.WELLNESS:
#         return {
#             "services_count": len(category_data.get("services_offered", [])),
#             "therapists_count": len(category_data.get("therapists", []))
#         }
#     elif main_category == MainCategory.SHOPPING:
#         return {
#             "products_count": len(category_data.get("inventory_items", [])),
#             "categories_count": len(category_data.get("product_categories", []))
#         }
#     elif main_category == MainCategory.ACTIVITIES:
#         return {
#             "max_group_size": category_data.get("group_size", {}).get("max", 0),
#             "equipment_provided": len(category_data.get("equipment_provided", []))
#         }
#     elif main_category == MainCategory.TRANSPORTATION:
#         return {
#             "vehicles_count": len(category_data.get("vehicle_types", [])),
#             "coverage_areas": len(category_data.get("coverage_areas", []))
#         }
    
#     return {}