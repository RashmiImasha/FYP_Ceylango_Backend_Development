
from fastapi import APIRouter, HTTPException, Depends
from firebase_admin import auth
from app.database.connection import (
    user_collection, 
    profiles_collection, 
    reviews_collection
)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

# Google Analytics imports
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import DateRange, Dimension, Metric, FilterExpression, Filter
from google.oauth2 import service_account
from app.config.settings import settings

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()





async def get_current_service_provider(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return current service provider"""
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

def calculate_profile_stats(profile_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dashboard statistics from profile data"""
    required_fields = ["service_name", "description", "address", "phone_number", "service_category"]
    required_completed = sum(1 for field in required_fields if profile_data.get(field))
    required_percentage = (required_completed / len(required_fields)) * 60 if required_fields else 0
    
    optional_score = 0
    if profile_data.get("profile_images"):
        optional_score += 15
    if profile_data.get("poster_images"):
        optional_score += 10
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

def get_all_service_reviews(uid: str):
    """Get all approved reviews for a service provider (index-friendly query)"""
    try:
        # Simple query - only filter by reviewable_id first
        reviews = reviews_collection\
            .where("reviewable_id", "==", uid)\
            .stream()
        
        reviews_list = []
        for doc in reviews:
            data = doc.to_dict()
            # Filter locally for service type and approved status
            if data.get("reviewable_type") == "service" and data.get("status") == "approved":
                reviews_list.append({
                    "id": doc.id,
                    **data,
                    "created_at": datetime.fromisoformat(data.get("created_at", "")) if data.get("created_at") else datetime.now()
                })
        
        return reviews_list
    except Exception as e:
        logger.error(f"Error getting service reviews for UID {uid}: {str(e)}")
        return []

def get_reviews_trend(uid: str, months: int = 6) -> list:
    """Get reviews trend data for the past N months"""
    trend_data = []
    
    # Get all reviews first using the simplified query
    all_reviews = get_all_service_reviews(uid)
    
    for i in range(months - 1, -1, -1):
        date = datetime.now() - timedelta(days=30 * i)
        month_name = date.strftime("%b")
        month_start = date.replace(day=1)
        
        if i > 0:
            month_end = (date.replace(day=1) + timedelta(days=32)).replace(day=1) - timedelta(days=1)
        else:
            month_end = datetime.now()
        
        # Filter locally by month
        positive_count = sum(
            1 for review in all_reviews
            if month_start <= review["created_at"] <= month_end and review.get("rating", 0) >= 4
        )
        negative_count = sum(
            1 for review in all_reviews
            if month_start <= review["created_at"] <= month_end and review.get("rating", 0) <= 2
        )
        
        trend_data.append({
            "month": month_name,
            "positive": positive_count,
            "negative": negative_count
        })
    
    return trend_data


def get_rating_distribution(uid: str) -> Dict[str, int]:
    """Get rating distribution breakdown"""
    rating_breakdown = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    
    # Use the simplified query
    all_reviews = get_all_service_reviews(uid)
    
    for review in all_reviews:
        rating = str(review.get("rating", 0))
        if rating in rating_breakdown:
            rating_breakdown[rating] += 1
    
    return rating_breakdown


def get_recent_reviews(uid: str, limit: int = 4) -> list:
    """Get recent reviews for the service provider"""
    all_reviews = get_all_service_reviews(uid)
    
    # Sort by created_at locally
    all_reviews.sort(key=lambda x: x.get("created_at", datetime.now()), reverse=True)
    
    # Take the most recent ones
    recent_reviews = all_reviews[:limit]
    
    # Format for frontend
    formatted_reviews = []
    for review in recent_reviews:
        formatted_reviews.append({
            "id": review.get("id"),
            "guest": review.get("user_name", "Anonymous"),
            "rating": review.get("rating", 0),
            "date": review.get("created_at").isoformat() if isinstance(review.get("created_at"), datetime) else review.get("created_at", ""),
            "comment": review.get("comment", ""),
            "title": review.get("title", ""),
            "replied": False
        })
    
    return formatted_reviews


@router.get("/dashboard/overview")
async def get_dashboard_overview(current_user: tuple = Depends(get_current_service_provider)):
    """Get complete dashboard overview with all statistics including profile views"""
    uid, user_data = current_user
    
    try:
        # Fetch profile data
        profile_doc = profiles_collection.document(uid).get()
        if not profile_doc.exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile_doc.to_dict()
        
        # Calculate completion stats
        completion_stats = calculate_profile_stats(profile_data)
        
        # Get reviews statistics
        all_reviews = get_all_service_reviews(uid)
        total_reviews = len(all_reviews)
        
        # Calculate average rating
        if total_reviews > 0:
            sum_ratings = sum(review.get("rating", 0) for review in all_reviews)
            average_rating = round(sum_ratings / total_reviews, 1)
        else:
            average_rating = profile_data.get("average_rating", 0.0)
        
        # Get profile views from profile data
        profile_views = profile_data.get("profile_views", 0)
        
        # Get trends and data
        reviews_trend = get_reviews_trend(uid)
        rating_breakdown = get_rating_distribution(uid)
        recent_reviews = get_recent_reviews(uid)
        
        # Convert rating breakdown to frontend format
        rating_distribution = [
            {"name": "5 Star", "value": rating_breakdown.get("5", 0)},
            {"name": "4 Star", "value": rating_breakdown.get("4", 0)},
            {"name": "3 Star", "value": rating_breakdown.get("3", 0)},
            {"name": "2 Star", "value": rating_breakdown.get("2", 0)},
            {"name": "1 Star", "value": rating_breakdown.get("1", 0)},
        ]
        
        return {
            "status": "success",
            "data": {
                "provider_info": {
                    "uid": uid,
                    "full_name": user_data.get("full_name", "Unknown"),
                    "email": user_data.get("email", ""),
                    "service_name": profile_data.get("service_name", "Unnamed Business"),
                    "service_category": profile_data.get("service_category", "Service"),
                    "district": profile_data.get("district", "Not set"),
                    "is_active": profile_data.get("is_active", True)
                },
                "profile_image": profile_data.get("profile_images", [None])[0] if profile_data.get("profile_images") else None,
                "metrics": {
                    "average_rating": average_rating,
                    "total_reviews": total_reviews,
                    "profile_views": profile_views,  # From Firestore profile
                    "profile_completion": completion_stats["total_percentage"]
                },
                "profile_completion": completion_stats,
                "quick_stats": {
                    "profile_images_count": len(profile_data.get("profile_images", [])),
                    "poster_images_count": len(profile_data.get("poster_images", [])),
                    "amenities_count": len(profile_data.get("amenities", [])),
                    "overall_rating": average_rating
                },
                "reviews_trend": reviews_trend,
                "rating_distribution": rating_distribution,
                "recent_reviews": recent_reviews,
                "profile_data": {
                    "service_name": profile_data.get("service_name"),
                    "description": profile_data.get("description"),
                    "address": profile_data.get("address"),
                    "phone_number": profile_data.get("phone_number"),
                    "website": profile_data.get("website"),
                    "operating_hours": profile_data.get("operating_hours"),
                    "amenities": profile_data.get("amenities", []),
                    "social_media": profile_data.get("social_media")
                }
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching dashboard overview for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")



@router.get("/dashboard/profile-completion-tasks")
async def get_profile_completion_tasks(current_user: tuple = Depends(get_current_service_provider)):
    """Get profile completion tasks"""
    uid, user_data = current_user
    
    try:
        profile_doc = profiles_collection.document(uid).get()
        if not profile_doc.exists:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile_doc.to_dict()
        completion_stats = calculate_profile_stats(profile_data)
        
        tasks = [
            {
                "task": "Add profile images",
                "completed": completion_stats["has_profile_images"],
                "weight": 15,
                "action": "photos"
            },
            {
                "task": "Add promotional posters",
                "completed": completion_stats["has_poster_images"],
                "weight": 10,
                "action": "posters"
            },
            {
                "task": "Set operating hours",
                "completed": completion_stats["has_operating_hours"],
                "weight": 5,
                "action": "hours"
            },
            {
                "task": "Add location coordinates",
                "completed": completion_stats["has_coordinates"],
                "weight": 5,
                "action": "location"
            },
            {
                "task": "Connect social media",
                "completed": completion_stats["has_social_media"],
                "weight": 3,
                "action": "social"
            },
            {
                "task": "List amenities/features",
                "completed": completion_stats["has_amenities"],
                "weight": 2,
                "action": "amenities"
            }
        ]
        
        logger.info(f"Profile completion tasks fetched for UID {uid}")
        
        return {
            "status": "success",
            "total_percentage": completion_stats["total_percentage"],
            "tasks": tasks
        }
    
    except Exception as e:
        logger.error(f"Error fetching completion tasks for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


