from fastapi import APIRouter, HTTPException
from app.database.connection import user_collection, destination_collection, profiles_collection, feedback_collection
from datetime import datetime, timedelta
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


def get_month_range(months_ago: int = 0):
    """Get start and end dates for a specific month"""
    today = datetime.now()
    if months_ago == 0:
        # Current month
        start = datetime(today.year, today.month, 1)
        end = today
    else:
        # Previous month
        target_date = today - timedelta(days=30 * months_ago)
        start = datetime(target_date.year, target_date.month, 1)
        # Last day of that month
        if target_date.month == 12:
            end = datetime(target_date.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = datetime(target_date.year, target_date.month + 1, 1) - timedelta(days=1)
    
    return start, end


def get_month_start_end(year: int, month: int):
    """Get first and last day of a specific month"""
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        end = datetime(year, month + 1, 1) - timedelta(days=1)
    return start, end


@router.get("/dashboard/stats")
async def get_dashboard_stats():
    """Get comprehensive dashboard statistics"""
    try:
        # Get current and previous month ranges
        current_start, current_end = get_month_range(0)
        prev_start, prev_end = get_month_range(1)
        
        # 1. TOTAL USERS (tourists only)
        all_users = list(user_collection.where("role", "==", "tourist").stream())
        total_users = len(all_users)
        
        # Count users from current and previous month
        current_month_users = sum(
            1 for doc in all_users
            if doc.to_dict().get("created_at") and 
            current_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= current_end
        )
        
        prev_month_users = sum(
            1 for doc in all_users
            if doc.to_dict().get("created_at") and 
            prev_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= prev_end
        )
        
        users_growth = calculate_growth(current_month_users, prev_month_users)
        
        # 2. TOTAL DESTINATIONS
        all_destinations = list(destination_collection.stream())
        total_destinations = len(all_destinations)
        
        current_month_destinations = sum(
            1 for doc in all_destinations
            if doc.to_dict().get("created_at") and 
            current_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= current_end
        )
        
        prev_month_destinations = sum(
            1 for doc in all_destinations
            if doc.to_dict().get("created_at") and 
            prev_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= prev_end
        )
        
        destinations_growth = calculate_growth(current_month_destinations, prev_month_destinations)
        
        # 3. ACTIVE SERVICES (service providers)
        all_services = list(profiles_collection.where("is_active", "==", True).stream())
        total_services = len(all_services)
        
        current_month_services = sum(
            1 for doc in all_services
            if doc.to_dict().get("created_at") and 
            current_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= current_end
        )
        
        prev_month_services = sum(
            1 for doc in all_services
            if doc.to_dict().get("created_at") and 
            prev_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= prev_end
        )
        
        services_growth = calculate_growth(current_month_services, prev_month_services)
        
        # 4. CONTENT FEEDBACK COUNT
        all_feedback = list(feedback_collection.stream())
        total_feedback = len(all_feedback)
        
        current_month_feedback = sum(
            1 for doc in all_feedback
            if doc.to_dict().get("created_at") and 
            current_start <= doc.to_dict()["created_at"].replace(tzinfo=None) <= current_end
        )
        
        prev_month_feedback = sum(
            1 for doc in all_feedback
            if doc.to_dict().get("created_at") and 
            prev_start <= doc.to_dict()["created_at"].replace(tzinfo=None) <= prev_end
        )
        
        feedback_growth = calculate_growth(current_month_feedback, prev_month_feedback)
        
        logger.info("Dashboard stats fetched successfully")
        
        return {
            "total_users": {
                "count": total_users,
                "growth": users_growth,
                "current_month": current_month_users,
                "previous_month": prev_month_users
            },
            "destinations": {
                "count": total_destinations,
                "growth": destinations_growth,
                "current_month": current_month_destinations,
                "previous_month": prev_month_destinations
            },
            "active_services": {
                "count": total_services,
                "growth": services_growth,
                "current_month": current_month_services,
                "previous_month": prev_month_services
            },
            "content_feedback": {
                "count": total_feedback,
                "growth": feedback_growth,
                "current_month": current_month_feedback,
                "previous_month": prev_month_feedback
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching dashboard stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard/content-growth")
async def get_content_growth():
    """Get content growth (Destinations and Services) for last 12 months"""
    try:
        destinations = list(destination_collection.stream())
        services = list(profiles_collection.stream())
        
        growth_data = []
        today = datetime.now()
        
        # Calculate 12 months ago
        twelve_months_ago = today.replace(day=1)
        for _ in range(11):
            if twelve_months_ago.month == 1:
                twelve_months_ago = twelve_months_ago.replace(year=twelve_months_ago.year - 1, month=12)
            else:
                twelve_months_ago = twelve_months_ago.replace(month=twelve_months_ago.month - 1)
        
        # Generate 12 month entries
        current_month = twelve_months_ago
        for _ in range(12):
            month_start, month_end = get_month_start_end(current_month.year, current_month.month)
            
            dest_count = sum(
                1 for doc in destinations
                if doc.to_dict().get("created_at") and
                month_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= month_end
            )
            
            service_count = sum(
                1 for doc in services
                if doc.to_dict().get("created_at") and
                month_start <= datetime.fromisoformat(doc.to_dict()["created_at"]) <= month_end
            )
            
            growth_data.append({
                "month": month_start.strftime("%b %Y"),
                "destinations": dest_count,
                "services": service_count
            })
            
            # Move to next month
            if current_month.month == 12:
                current_month = current_month.replace(year=current_month.year + 1, month=1)
            else:
                current_month = current_month.replace(month=current_month.month + 1)
        
        logger.info("Content growth data fetched successfully")
        return {"growth": growth_data}
        
    except Exception as e:
        logger.error(f"Error fetching content growth: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def calculate_growth(current: int, previous: int) -> Dict[str, Any]:
    """Calculate growth percentage and direction"""
    if previous == 0:
        if current > 0:
            return {
                "percentage": 100.0,
                "direction": "up",
                "is_growth": True
            }
        return {
            "percentage": 0.0,
            "direction": "neutral",
            "is_growth": False
        }
    
    change = current - previous
    percentage = (change / previous) * 100
    
    return {
        "percentage": abs(round(percentage, 1)),
        "direction": "up" if change > 0 else "down" if change < 0 else "neutral",
        "is_growth": change > 0
    }