from fastapi import APIRouter, Query, HTTPException
from app.database.connection import db
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

destinations_collection = db.collection("destination")
profiles_collection = db.collection("service_provider_profiles")
reviews_collection = db.collection("reviews")


def calculate_wilson_score(positive: int, total: int) -> float:
    """Calculate Wilson score for better ranking with varying review counts"""
    if total == 0:
        return 0
    
    z = 1.96  # 95% confidence
    phat = positive / total
    
    score = (phat + z*z/(2*total) - z * ((phat*(1-phat)+z*z/(4*total))/total)**0.5) / (1+z*z/total)
    return score


def calculate_bayesian_average(rating: float, count: int, global_avg: float = 3.5, confidence: int = 10) -> float:
    """Calculate Bayesian average to handle items with few reviews"""
    return (confidence * global_avg + count * rating) / (confidence + count)


async def get_recent_reviews_count(reviewable_type: str, reviewable_id: str, cutoff_date: str) -> int:
    """Get count of recent reviews with simpler query to avoid composite indexes"""
    try:
        # First filter by type and ID (this should use existing index)
        base_query = reviews_collection\
            .where("reviewable_type", "==", reviewable_type)\
            .where("reviewable_id", "==", reviewable_id)\
            .where("status", "==", "approved")
        
        # Then filter in code (less efficient but avoids composite index)
        reviews = base_query.stream()
        recent_count = 0
        for review in reviews:
            review_data = review.to_dict()
            created_at = review_data.get("created_at")
            if created_at and created_at >= cutoff_date:
                recent_count += 1
        
        return recent_count
    except Exception as e:
        logger.warning(f"Error counting recent reviews: {str(e)}")
        return 0


@router.get("/destinations/popular")
async def get_popular_destinations(
    limit: int = Query(10, le=50),
    days: int = Query(90, description="Consider reviews from last N days for popularity")
):
    """
    Get popular destinations with optimized queries to avoid composite indexes
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
            
            # Get recent reviews count with optimized query
            recent_count = await get_recent_reviews_count("destination", dest_id, cutoff_date)
            
            # Calculate popularity score
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
    """Get top-rated destinations using Bayesian average - no complex queries needed"""
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
    Get trending destinations - simplified to avoid composite indexes
    """
    try:
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get destinations
        destinations = destinations_collection.stream()
        
        trending_list = []
        
        for doc in destinations:
            dest_data = doc.to_dict()
            dest_id = doc.id
            
            # Get recent reviews count with optimized query
            recent_count = await get_recent_reviews_count("destination", dest_id, cutoff_date)
            
            # Skip if not enough recent activity
            if recent_count < 3:
                continue
            
            avg_rating = dest_data.get("average_rating", 0)
            
            # Trending score: combines recent activity with quality
            trending_score = (
                recent_count * 0.6 +  # More weight on activity
                avg_rating * 2 * 0.4  # Some weight on quality (scaled to match)
            )
            
            dest_data["id"] = dest_id
            dest_data["trending_score"] = round(trending_score, 2)
            dest_data["recent_reviews_count"] = recent_count
            dest_data["recent_average_rating"] = round(avg_rating, 2)  # Using overall avg as approximation
            
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


# Update the services endpoints similarly...

@router.get("/services/popular")
async def get_popular_services(
    service_category: Optional[str] = Query(None),
    limit: int = Query(10, le=50),
    days: int = Query(90)
):
    """Get popular service providers with optimized queries"""
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
            
            # Get recent reviews count with optimized query
            recent_count = await get_recent_reviews_count("service", service_id, cutoff_date)
            
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