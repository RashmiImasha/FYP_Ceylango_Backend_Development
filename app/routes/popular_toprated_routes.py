from fastapi import APIRouter, Query, HTTPException  # Fixed import
from app.database.connection import db
from datetime import datetime, timedelta
from typing import Literal, Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

destinations_collection = db.collection("destination")
profiles_collection = db.collection("service_provider_profiles")
reviews_collection = db.collection("reviews")


def calculate_wilson_score(positive: int, total: int) -> float:
    """
    Calculate Wilson score for better ranking with varying review counts
    This prevents items with 1 five-star review from ranking above items with 100 four-star reviews
    """
    if total == 0:
        return 0
    
    z = 1.96  # 95% confidence
    phat = positive / total
    
    score = (phat + z*z/(2*total) - z * ((phat*(1-phat)+z*z/(4*total))/total)**0.5) / (1+z*z/total)
    return score


def calculate_bayesian_average(rating: float, count: int, global_avg: float = 3.5, confidence: int = 10) -> float:
    """
    Calculate Bayesian average to handle items with few reviews
    Formula: (C * m + R * v) / (C + v)
    where C = confidence, m = global average, R = count, v = average rating
    """
    return (confidence * global_avg + count * rating) / (confidence + count)


@router.get("/destinations/popular")
async def get_popular_destinations(
    limit: int = Query(10, le=50),
    days: int = Query(90, description="Consider reviews from last N days for popularity")
):
    """
    Get popular destinations based on:
    - Average rating
    - Number of reviews
    - Recent activity (reviews in last 90 days)
    """
    try:
        # Get all destinations with ratings
        destinations = destinations_collection.stream()
        
        popular_list = []
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        for doc in destinations:
            dest_data = doc.to_dict()
            dest_id = doc.id
            
            avg_rating = dest_data.get("average_rating", 0)
            total_reviews = dest_data.get("total_reviews", 0)
            
            # Skip if not enough reviews
            if total_reviews < 3:
                continue
            
            # Get recent reviews count
            recent_reviews = reviews_collection\
                .where("reviewable_type", "==", "destination")\
                .where("reviewable_id", "==", dest_id)\
                .where("status", "==", "approved")\
                .where("created_at", ">=", cutoff_date)\
                .stream()
            
            recent_count = len(list(recent_reviews))
            
            # Calculate popularity score
            # Combines: Bayesian average, review count, and recent activity
            bayesian_avg = calculate_bayesian_average(avg_rating, total_reviews)
            popularity_score = (
                bayesian_avg * 0.5 +  # 50% weight on quality
                min(total_reviews / 50, 1) * 0.3 +  # 30% weight on review count (capped)
                min(recent_count / 10, 1) * 0.2  # 20% weight on recent activity
            ) * 5  # Scale to 0-5
            
            dest_data["id"] = dest_id
            dest_data["popularity_score"] = round(popularity_score, 2)
            dest_data["recent_reviews_count"] = recent_count
            
            popular_list.append(dest_data)
        
        # Sort by popularity score
        popular_list.sort(key=lambda x: x["popularity_score"], reverse=True)
        logger.info(f"Fetched popular destinations, total found: {len(popular_list)}")        
        return {
            "count": min(len(popular_list), limit),
            "destinations": popular_list[:limit],
            "calculation_period_days": days
        }
    
    except Exception as e:
        logger.error(f"Error fetching popular destinations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/destinations/top-rated")
async def get_top_rated_destinations(
    limit: int = Query(10, le=50),
    min_reviews: int = Query(5, description="Minimum number of reviews required")
):
    """Get top-rated destinations using Bayesian average to handle varying review counts"""
    try:
        destinations = destinations_collection.stream()
        
        rated_list = []
        
        for doc in destinations:
            dest_data = doc.to_dict()
            
            avg_rating = dest_data.get("average_rating", 0)
            total_reviews = dest_data.get("total_reviews", 0)
            
            # Filter by minimum reviews
            if total_reviews < min_reviews:
                continue
            
            # Calculate weighted rating using Bayesian average
            weighted_rating = calculate_bayesian_average(avg_rating, total_reviews)
            
            dest_data["id"] = doc.id
            dest_data["weighted_rating"] = round(weighted_rating, 2)
            
            rated_list.append(dest_data)
        
        # Sort by weighted rating
        rated_list.sort(key=lambda x: (x["weighted_rating"], x["total_reviews"]), reverse=True)
        logger.info(f"Fetched top-rated destinations, total found: {len(rated_list)}")
        
        return {
            "count": min(len(rated_list), limit),
            "destinations": rated_list[:limit],
            "min_reviews_filter": min_reviews
        }
    
    except Exception as e:
        logger.error(f"Error fetching top-rated destinations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/destinations/trending")
async def get_trending_destinations(
    limit: int = Query(10, le=50),
    days: int = Query(30, description="Look at reviews from last N days")
):
    """
    Get trending destinations based on recent review activity
    Trending = high number of recent reviews + good average rating
    """
    try:
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get destinations
        destinations = destinations_collection.stream()
        
        trending_list = []
        
        for doc in destinations:
            dest_data = doc.to_dict()
            dest_id = doc.id
            
            # Get recent reviews
            recent_reviews_query = reviews_collection\
                .where("reviewable_type", "==", "destination")\
                .where("reviewable_id", "==", dest_id)\
                .where("status", "==", "approved")\
                .where("created_at", ">=", cutoff_date)\
                .stream()
            
            recent_reviews_list = list(recent_reviews_query)
            recent_count = len(recent_reviews_list)
            
            # Skip if not enough recent activity
            if recent_count < 3:
                continue
            
            # Calculate recent average rating
            recent_avg = sum(r.to_dict().get("rating", 0) for r in recent_reviews_list) / recent_count
            
            # Trending score: combines recent activity with quality
            trending_score = (
                recent_count * 0.6 +  # More weight on activity
                recent_avg * 2 * 0.4  # Some weight on quality (scaled to match)
            )
            
            dest_data["id"] = dest_id
            dest_data["trending_score"] = round(trending_score, 2)
            dest_data["recent_reviews_count"] = recent_count
            dest_data["recent_average_rating"] = round(recent_avg, 2)
            
            trending_list.append(dest_data)
        
        # Sort by trending score
        trending_list.sort(key=lambda x: x["trending_score"], reverse=True)
        logger.info(f"Fetched trending destinations, total found: {len(trending_list)}")
        
        return {
            "count": min(len(trending_list), limit),
            "destinations": trending_list[:limit],
            "period_days": days
        }
    
    except Exception as e:
        logger.error(f"Error fetching trending destinations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/top-rated")
async def get_top_rated_services(
    service_category: Optional[str] = Query(None),
    limit: int = Query(10, le=50),
    min_reviews: int = Query(5)
):
    """Get top-rated service providers"""
    try:
        query = profiles_collection.where("is_active", "==", True)
        
        if service_category:
            query = query.where("service_category", "==", service_category)
        
        services = query.stream()
        
        rated_list = []
        
        for doc in services:
            service_data = doc.to_dict()
            
            avg_rating = service_data.get("average_rating", 0)
            total_reviews = service_data.get("total_reviews", 0)
            
            if total_reviews < min_reviews:
                continue
            
            # Calculate weighted rating
            weighted_rating = calculate_bayesian_average(avg_rating, total_reviews)
            
            service_data["uid"] = doc.id
            service_data["weighted_rating"] = round(weighted_rating, 2)
            
            rated_list.append(service_data)
        
        # Sort by weighted rating
        rated_list.sort(key=lambda x: (x["weighted_rating"], x["total_reviews"]), reverse=True)
        logger.info(f"Fetched top-rated services, total found: {len(rated_list)}")
        
        return {
            "count": min(len(rated_list), limit),
            "services": rated_list[:limit],
            "filters": {
                "service_category": service_category,
                "min_reviews": min_reviews
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching top-rated services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/popular")
async def get_popular_services(
    service_category: Optional[str] = Query(None),
    limit: int = Query(10, le=50),
    days: int = Query(90)
):
    """Get popular service providers"""
    try:
        query = profiles_collection.where("is_active", "==", True)
        
        if service_category:
            query = query.where("service_category", "==", service_category)
        
        services = query.stream()
        
        popular_list = []
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        for doc in services:
            service_data = doc.to_dict()
            service_id = doc.id
            
            avg_rating = service_data.get("average_rating", 0)
            total_reviews = service_data.get("total_reviews", 0)
            
            if total_reviews < 3:
                continue
            
            # Get recent reviews
            recent_reviews = reviews_collection\
                .where("reviewable_type", "==", "service")\
                .where("reviewable_id", "==", service_id)\
                .where("status", "==", "approved")\
                .where("created_at", ">=", cutoff_date)\
                .stream()
            
            recent_count = len(list(recent_reviews))
            
            # Calculate popularity score
            bayesian_avg = calculate_bayesian_average(avg_rating, total_reviews)
            popularity_score = (
                bayesian_avg * 0.5 +
                min(total_reviews / 50, 1) * 0.3 +
                min(recent_count / 10, 1) * 0.2
            ) * 5
            
            service_data["uid"] = service_id
            service_data["popularity_score"] = round(popularity_score, 2)
            service_data["recent_reviews_count"] = recent_count
            
            popular_list.append(service_data)
        
        popular_list.sort(key=lambda x: x["popularity_score"], reverse=True)
        logger.info(f"Fetched popular services, total found: {len(popular_list)}")
        return {
            "count": min(len(popular_list), limit),
            "services": popular_list[:limit],
            "filters": {
                "service_category": service_category
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching popular services: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/destinations/{destination_id}/stats")
async def get_destination_rating_stats(destination_id: str):
    """Get detailed rating statistics for a destination"""
    try:
        dest_doc = destinations_collection.document(destination_id).get()
        
        if not dest_doc.exists:
            raise HTTPException(status_code=404, detail="Destination not found")
        
        dest_data = dest_doc.to_dict()
        logger.info(f"Fetched rating stats for destination ID: {destination_id}")
        
        return {
            "destination_id": destination_id,
            "destination_name": dest_data.get("destination_name"),
            "average_rating": dest_data.get("average_rating", 0),
            "total_reviews": dest_data.get("total_reviews", 0),
            "rating_breakdown": dest_data.get("rating_breakdown", {
                "5": 0, "4": 0, "3": 0, "2": 0, "1": 0
            })
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching destination rating stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/services/{service_id}/stats")
async def get_service_rating_stats(service_id: str):
    """Get detailed rating statistics for a service provider"""
    try:
        service_doc = profiles_collection.document(service_id).get()
        
        if not service_doc.exists:
            raise HTTPException(status_code=404, detail="Service not found")
        
        service_data = service_doc.to_dict()
        logger.info(f"Fetched rating stats for service ID: {service_id}")
        
        return {
            "service_id": service_id,
            "service_name": service_data.get("service_name"),
            "average_rating": service_data.get("average_rating", 0),
            "total_reviews": service_data.get("total_reviews", 0),
            "rating_breakdown": service_data.get("rating_breakdown", {
                "5": 0, "4": 0, "3": 0, "2": 0, "1": 0
            })
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching service rating stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))