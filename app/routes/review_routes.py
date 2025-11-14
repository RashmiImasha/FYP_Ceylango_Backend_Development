from fastapi import APIRouter, HTTPException, status, Query
from firebase_admin import auth
from app.database.connection import destination_collection, reviews_collection, profiles_collection, user_collection
from app.models.review import HelpfulRequest, ReviewCreate, ReviewUpdate
from fastapi.security import HTTPBearer
from datetime import datetime
from typing import Optional, List, Literal
import uuid, logging

logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer()


# ===== EMAIL-BASED AUTH DEPENDENCY =====
async def get_current_user_by_email(user_email: str):
    """Verify user exists by email and return user data"""
    try:
        # Find user by email in user_collection
        users = user_collection.where("email", "==", user_email.lower().strip()).limit(1).get()
        user_list = list(users)
        
        if not user_list:
            logger.warning(f"User not found. Please register first. Email: {user_email}")
            raise HTTPException(status_code=404, detail="User not found. Please register first.")
        
        user_doc = user_list[0]
        user_data = user_doc.to_dict()
        uid = user_doc.id

        logger.info(f"Authenticated user by email: {user_email}, UID: {uid}")        
        return uid, user_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error for email {user_email}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Authentication error: {str(e)}")



# ===== HELPER FUNCTIONS =====
def verify_reviewable_exists(reviewable_type: str, reviewable_id: str):
    """Verify that the destination or service exists"""
    if reviewable_type == "destination":
        doc = destination_collection.document(reviewable_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Destination not found")
    elif reviewable_type == "service":
        doc = profiles_collection.document(reviewable_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Service provider not found")
    else:
        logger.warning(f"Invalid reviewable type: {reviewable_type}")
        raise HTTPException(status_code=400, detail="Invalid reviewable type")


def check_existing_review(user_id: str, reviewable_type: str, reviewable_id: str):
    """Check if user has already reviewed this item"""
    existing = reviews_collection\
        .where("user_id", "==", user_id)\
        .where("reviewable_type", "==", reviewable_type)\
        .where("reviewable_id", "==", reviewable_id)\
        .limit(1)\
        .get()
    
    return len(list(existing)) > 0


def update_aggregated_ratings(reviewable_type: str, reviewable_id: str):
    """Update average rating and total reviews count"""
    try:
        # Get all approved reviews
        reviews = reviews_collection\
            .where("reviewable_type", "==", reviewable_type)\
            .where("reviewable_id", "==", reviewable_id)\
            .where("status", "==", "approved")\
            .stream()
        
        total_reviews = 0
        sum_ratings = 0
        rating_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}  # ✅ String keys
        
        for review in reviews:
            review_data = review.to_dict()
            rating = review_data.get("rating", 0)
            sum_ratings += rating
            total_reviews += 1
            rating_counts[str(rating)] += 1  # ✅ Convert to string
        
        avg_rating = round(sum_ratings / total_reviews, 2) if total_reviews > 0 else 0.0
        
        # Update the destination or service document
        collection = destination_collection if reviewable_type == "destination" else profiles_collection
        doc_ref = collection.document(reviewable_id)
        
        doc_ref.update({
            "average_rating": avg_rating,
            "total_reviews": total_reviews,
            "rating_breakdown": rating_counts,
            "updated_at": datetime.now().isoformat()
        })
        logger.info(f"Updated aggregated ratings for {reviewable_type} ID {reviewable_id}: Avg Rating {avg_rating}, Total Reviews {total_reviews}")
        return avg_rating, total_reviews
    
    except Exception as e:
        logger.error(f"Error updating aggregated ratings: {str(e)}")
        # Don't fail the main operation if rating update fails
        return None, None


@router.post("/reviews", status_code=status.HTTP_201_CREATED)
async def create_review(
    review_data: ReviewCreate
):
    """Create a new review using email authentication"""
    try:
        # Authenticate user by email
        uid, user_data = await get_current_user_by_email(review_data.user_email)
        
        # Validate reviewable exists
        verify_reviewable_exists(review_data.reviewable_type, review_data.reviewable_id)
        
        
        # Check if user already reviewed
        if check_existing_review(uid, review_data.reviewable_type, review_data.reviewable_id):
            raise HTTPException(
                status_code=400, 
                detail="You have already reviewed this item"
            )
        
        # Create review document
        review_id = str(uuid.uuid4())
        review_doc_data = {
            "user_id": uid,
            "user_name": user_data.get("full_name", "Anonymous"),
            "user_email": user_data.get("email", ""),
            "reviewable_type": review_data.reviewable_type,
            "reviewable_id": review_data.reviewable_id,
            "rating": review_data.rating,
            "title": review_data.title.strip(),
            "comment": review_data.comment.strip(),
            "visit_date": review_data.visit_date,
            "helpful_count": 0,
            "helpful_by": [],
            "is_verified": False,
            "status": "approved",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        reviews_collection.document(review_id).set(review_doc_data)
        
        # Update aggregated ratings
        update_aggregated_ratings(review_data.reviewable_type, review_data.reviewable_id)
        
        review_doc_data["id"] = review_id
        logger.info(f"Created new review ID {review_id} by user {uid} for {review_data.reviewable_type} ID {review_data.reviewable_id}")
        return {
            "message": "Review created successfully",
            "review": review_doc_data
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/reviews/{review_id}", response_model=dict)
async def get_review(review_id: str):
    """Get a single review by ID"""
    try:
        review_doc = reviews_collection.document(review_id).get()
        
        if not review_doc.exists:
            raise HTTPException(status_code=404, detail="Review not found")
        
        review_data = review_doc.to_dict()
        review_data["id"] = review_id

        logger.info(f"Fetched review ID {review_id}")        
        return {"review": review_data}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching review ID {review_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/reviews/{review_id}")
async def update_review(
    review_id: str,
    review_update: ReviewUpdate,
    user_email: str = Query(..., description="User email for authentication")
):
    """Update an existing review"""
    uid, user_data = await get_current_user_by_email(user_email)
    
    try:
        review_doc = reviews_collection.document(review_id)
        review_snapshot = review_doc.get()
        
        if not review_snapshot.exists:
            raise HTTPException(status_code=404, detail="Review not found")
        
        review_data = review_snapshot.to_dict()
        
        # Check if user owns this review
        if review_data.get("user_id") != uid:
            raise HTTPException(status_code=403, detail="Not authorized to update this review")
        
        # Prepare update data
        update_data = {"updated_at": datetime.now().isoformat()}
        
        # Only update fields that are provided (not None)
        if review_update.rating is not None:
            update_data["rating"] = review_update.rating
        if review_update.title is not None:
            update_data["title"] = review_update.title.strip()
        if review_update.comment is not None:
            update_data["comment"] = review_update.comment.strip()
        if review_update.visit_date is not None:
            update_data["visit_date"] = review_update.visit_date
        
        # Update review
        review_doc.update(update_data)
        
        # Update aggregated ratings if rating changed
        if review_update.rating is not None:
            update_aggregated_ratings(
                review_data.get("reviewable_type"),
                review_data.get("reviewable_id")
            )
        
        # Get updated review
        updated_review = review_doc.get().to_dict()
        updated_review["id"] = review_id

        logger.info(f"Updated review ID {review_id} by user {uid}")        
        return {
            "message": "Review updated successfully",
            "review": updated_review
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating review ID {review_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/reviews/{review_id}")
async def delete_review(
    review_id: str,
    user_email: str = Query(..., description="User email for authentication")
):
    """Delete a review"""
    uid, user_data = await get_current_user_by_email(user_email)
    
    try:
        review_doc = reviews_collection.document(review_id)
        review_snapshot = review_doc.get()
        
        if not review_snapshot.exists:
            raise HTTPException(status_code=404, detail="Review not found")
        
        review_data = review_snapshot.to_dict()
        
        # Check if user owns this review or is admin
        if review_data.get("user_id") != uid and user_data.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Not authorized to delete this review")
        
        # Delete review
        review_doc.delete()
        
        # Update aggregated ratings
        update_aggregated_ratings(
            review_data.get("reviewable_type"),
            review_data.get("reviewable_id")
        )
        logger.info(f"Deleted review ID {review_id} by user {uid}")
        return {"message": "Review deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting review ID {review_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reviews/{review_id}/helpful")
async def mark_review_helpful(
    review_id: str,
    helpful_request: HelpfulRequest
):
    """Mark a review as helpful (toggle) using request body"""
    try:
        # Authenticate user by email
        uid, user_data = await get_current_user_by_email(helpful_request.user_email)
        
        review_doc = reviews_collection.document(review_id)
        review_snapshot = review_doc.get()
        
        if not review_snapshot.exists:
            raise HTTPException(status_code=404, detail="Review not found")
        
        review_data = review_snapshot.to_dict()
        helpful_by = review_data.get("helpful_by", [])
        
        if uid in helpful_by:
            # Remove helpful vote
            helpful_by.remove(uid)
            message = "Helpful vote removed"
        else:
            # Add helpful vote
            helpful_by.append(uid)
            message = "Marked as helpful"
        
        review_doc.update({
            "helpful_by": helpful_by,
            "helpful_count": len(helpful_by),
            "updated_at": datetime.now().isoformat()
        })
        logger.info(f"User {uid} toggled helpful vote on review ID {review_id}")
        return {
            "message": message,
            "helpful_count": len(helpful_by)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error marking review ID {review_id} as helpful: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@router.get("/destinations/{destination_id}/reviews")
async def get_destination_reviews(
    destination_id: str,
    sort_by: Literal["recent", "helpful", "rating_high", "rating_low"] = Query("recent"),
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all reviews for a destination with filtering, sorting, and rating summary"""
    try:
        # First, get the destination to get current aggregated ratings
        dest_doc = destination_collection.document(destination_id).get()
        if not dest_doc.exists:
            raise HTTPException(status_code=404, detail="Destination not found")
        
        dest_data = dest_doc.to_dict()
        
        # Get reviews with filtering
        query = reviews_collection\
            .where("reviewable_type", "==", "destination")\
            .where("reviewable_id", "==", destination_id)\
            .where("status", "==", "approved")
        
        if min_rating:
            query = query.where("rating", ">=", min_rating)
        
        # Get all matching reviews
        reviews = list(query.stream())
        
        # Convert to list of dicts
        reviews_list = []
        for doc in reviews:
            review_data = doc.to_dict()
            review_data["id"] = doc.id
            reviews_list.append(review_data)
        
        # Sort reviews
        if sort_by == "recent":
            reviews_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "helpful":
            reviews_list.sort(key=lambda x: x.get("helpful_count", 0), reverse=True)
        elif sort_by == "rating_high":
            reviews_list.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "rating_low":
            reviews_list.sort(key=lambda x: x.get("rating", 0))
        
        # Apply pagination
        total_count = len(reviews_list)
        paginated_reviews = reviews_list[offset:offset + limit]
        
        # Prepare rating summary
        rating_breakdown = dest_data.get("rating_breakdown", {})
        rating_summary = {
            "average_rating": dest_data.get("average_rating", 0.0),
            "total_reviews": dest_data.get("total_reviews", 0),
            "breakdown": {
                "5": rating_breakdown.get("5", 0),  # Excellent
                "4": rating_breakdown.get("4", 0),  # Good
                "3": rating_breakdown.get("3", 0),  # Average
                "2": rating_breakdown.get("2", 0),  # Poor
                "1": rating_breakdown.get("1", 0),  # Terrible
            }
        }

        logger.info(f"Fetched reviews for destination ID {destination_id} with {len(paginated_reviews)} reviews returned")        
        return {
            "rating_summary": rating_summary,
            "total_count": total_count,
            "count": len(paginated_reviews),
            "reviews": paginated_reviews,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching reviews for destination ID {destination_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    
@router.get("/services/{service_id}/reviews")
async def get_service_reviews(
    service_id: str,
    sort_by: Literal["recent", "helpful", "rating_high", "rating_low"] = Query("recent"),
    min_rating: Optional[int] = Query(None, ge=1, le=5),
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all reviews for a service provider with filtering and sorting"""
    try:
        query = reviews_collection\
            .where("reviewable_type", "==", "service")\
            .where("reviewable_id", "==", service_id)\
            .where("status", "==", "approved")
        
        if min_rating:
            query = query.where("rating", ">=", min_rating)
        
        # Get all matching reviews
        reviews = list(query.stream())
        
        # Convert to list of dicts
        reviews_list = []
        for doc in reviews:
            review_data = doc.to_dict()
            review_data["id"] = doc.id
            reviews_list.append(review_data)
        
        # Sort reviews
        if sort_by == "recent":
            reviews_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        elif sort_by == "helpful":
            reviews_list.sort(key=lambda x: x.get("helpful_count", 0), reverse=True)
        elif sort_by == "rating_high":
            reviews_list.sort(key=lambda x: x.get("rating", 0), reverse=True)
        elif sort_by == "rating_low":
            reviews_list.sort(key=lambda x: x.get("rating", 0))
        
        # Apply pagination
        total_count = len(reviews_list)
        paginated_reviews = reviews_list[offset:offset + limit]
        
        logger.info(f"Fetched reviews for service ID {service_id} with {len(paginated_reviews)} reviews returned")
        return {
            "total_count": total_count,
            "count": len(paginated_reviews),
            "reviews": paginated_reviews,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total_count
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching reviews for service ID {service_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}/reviews")
async def get_user_reviews(
    user_id: str,
    limit: int = Query(20, le=100),
    offset: int = Query(0)
):
    """Get all reviews by a specific user"""
    try:
        reviews = reviews_collection\
            .where("user_id", "==", user_id)\
            .order_by("created_at", direction="DESCENDING")\
            .limit(limit)\
            .offset(offset)\
            .stream()
        
        reviews_list = []
        for doc in reviews:
            review_data = doc.to_dict()
            review_data["id"] = doc.id
            
            # Add reviewable details
            reviewable_type = review_data.get("reviewable_type")
            reviewable_id = review_data.get("reviewable_id")
            
            if reviewable_type == "destination":
                dest_doc = destination_collection.document(reviewable_id).get()
                if dest_doc.exists:
                    review_data["reviewable_name"] = dest_doc.to_dict().get("destination_name")
            elif reviewable_type == "service":
                service_doc = profiles_collection.document(reviewable_id).get()
                if service_doc.exists:
                    review_data["reviewable_name"] = service_doc.to_dict().get("service_name")
            
            reviews_list.append(review_data)
        
        logger.info(f"Fetched reviews by user ID {user_id} with {len(reviews_list)} reviews returned")        
        return {
            "count": len(reviews_list),
            "reviews": reviews_list,
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": len(reviews_list) == limit
            }
        }
    
    except Exception as e:
        logger.error(f"Error fetching reviews by user ID {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# @router.get("/reviews/my-reviews")
# async def get_my_reviews(
#     limit: int = Query(20, le=100),
#     offset: int = Query(0),
#     current_user: tuple = Depends(get_current_user)
# ):
#     """Get current user's reviews"""
#     uid, user_data = current_user
#     return await get_user_reviews(uid, limit, offset)