from fastapi import APIRouter, HTTPException, Query
from app.models.tripPlan import TripPlanRequest, TripPlanResponse
from app.database.connection import tripPlan_collection
from app.services.tripPlan_service import generate_trip_plan
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/create")
async def create_trip_plan(request: TripPlanRequest):
    try:
        result = await generate_trip_plan(request)
        return result
    except Exception as e:
        logger.error(f"Error creating trip plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/get/{trip_id}", response_model=TripPlanResponse)
async def get_trip_plan(trip_id: str):
    
    try:
        doc = tripPlan_collection.document(trip_id).get()        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Trip plan not found")
        
        data = doc.to_dict()
        response_data = {
            "trip_id": data["trip_id"],
            "trip_name": data["trip_name"],
            "summary": data["summary"],
            "itinerary": data["itinerary"],
            "map_data": data["map_data"],
            "generated_at": data.get("generated_at", ""),
            "alternatives": data.get("alternatives")
        }
        return TripPlanResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving trip plan: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all", response_model=List[TripPlanResponse])
async def get_all_trip_plans(
    limit: int = Query(default=10, le=100, description="Maximum number of trips to return"),
): 
    try:                   
        docs = tripPlan_collection.limit(limit).get()
        
        trip_plans = []
        for doc in docs:
            data = doc.to_dict()                        
            
            trip = TripPlanResponse(
                trip_id= data.get("trip_id"),
                trip_name= data.get("trip_name"),
                summary= data.get("summary"),
                itinerary= data.get("itinerary"),
                map_data= data.get("map_data"),
                generated_at= data.get("generated_at", ""),
                alternatives= data.get("alternatives")
            )
            trip_plans.append(trip)
        
        trip_plans.sort(
            key=lambda x: x.generated_at or "", 
            reverse=True
        )        
        return trip_plans
        
    except Exception as e:
        logger.error(f"Error fetching trip plans: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/delete/{trip_id}")
async def delete_trip_plan(trip_id: str, user_id: str):
    
    try:
        doc_ref = tripPlan_collection.document(trip_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Trip plan not found")
        
        data = doc.to_dict()
        if data['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized to delete this trip")
        
        doc_ref.delete()        
        return {"message": "Trip plan deleted successfully", "trip_id": trip_id}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/rename/{trip_id}")
async def rename_trip_plan(
    trip_id: str, 
    user_id: str, 
    new_name: str = Query(..., min_length=3, max_length=100)
):
    try:
        doc_ref = tripPlan_collection.document(trip_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Trip plan not found")
        
        data = doc.to_dict()
        if data['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Unauthorized")
        
        doc_ref.update({"trip_name": new_name})
        
        return {
            "message": "Trip renamed successfully",
            "trip_id": trip_id,
            "new_name": new_name
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error renaming trip: {e}")
        raise HTTPException(status_code=500, detail=str(e))

