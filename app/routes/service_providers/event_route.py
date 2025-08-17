from fastapi import APIRouter, HTTPException, UploadFile, status, Form, File
from app.models.service_providers.events import EventResponse
from app.utils.storage_handle import upload_file_to_storage, delete_file_from_storage, update_file_in_storage
from app.database.connection import event_collection
from typing import List, Optional
from pydantic import EmailStr
from datetime import date as DateType, time as TimeType

router = APIRouter()

# add event
@router.post("/", response_model=EventResponse)
def add_event(
    event_name: str = Form(...),
    date: DateType = Form(...),
    time: TimeType = Form(...),
    venue: str = Form(...),
    event_lat: float = Form(...),
    event_lon: float = Form(...),
    description: str = Form(...),
    post: UploadFile = File(...),
    event_image: List[UploadFile] = File(default=[]),
    event_video: List[UploadFile] = File(default=[]),
    email: EmailStr = Form(...)
):
    event_image = [img for img in event_image if img.filename]
    if not event_image:
        event_image = None

    event_video = [vid for vid in event_video if vid.filename]
    if not event_video:
        event_video = None

    existing_event = (
        event_collection
        .where("event_name", "==", event_name) 
        .where("venue", "==", venue)
        .where("date", "==", str(date)) 
        .where("time", "==", str(time)) 
        .get()
    )  
        
    if len(existing_event) > 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="An event with the same name, date, time and venue already exists.")
            
    # upload post
    post_url = upload_file_to_storage(post, "event_posts")

    # upload event image array
    event_img_urls = []
    if event_image:
        for img in event_image:
            url = upload_file_to_storage(img, "event_images")
            if url:
                event_img_urls.append(url)
    
    # upload event video array
    video_urls = []
    if event_video:
        for vid in event_video:
            url = upload_file_to_storage(vid, "event_videos")
            if url:
                video_urls.append(url)

    event_data = {
        "event_name" : event_name,
        "date" : date,
        "time" : time,
        "venue" : venue,
        "event_lat" : event_lat,
        "event_lon" : event_lon,
        "description" : description,
        "post" : post_url,
        "event_image" : event_img_urls or None,
        "event_video" : video_urls or None,
        "email" : email
    }

    event_dict = event_data.copy()
    event_dict['date'] = event_data['date'].isoformat()
    event_dict['time'] = event_data['time'].strftime("%H:%M:%S")

    _, doc_ref = event_collection.add(event_dict)

    return EventResponse (
        **event_dict,
        id = doc_ref.id
    )

# get event by Id
@router.get("/{event_id}", response_model=EventResponse)
def get_event_byId(event_id: str):
    doc_ref = event_collection.document(event_id)
    event = doc_ref.get()

    if not event.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evet is not found"
        )
    
    event_data = event.to_dict()
    event_data["id"] = event.id
    return event_data

# get all events
@router.get("/event/all", response_model=list[EventResponse])
def get_all_Event():
    events = event_collection.stream()
    result = []

    for doc in events:
        data = doc.to_dict()
        data['id'] = doc.id
        result.append(data)

    return result

# get all upcoming events
@router.get("/", response_model=list[EventResponse])
def get_upcoming_event():

    today = DateType.today().isoformat()
    events = event_collection.where("date", ">=", today).stream()
    
    result = []
    for doc in events:
        data = doc.to_dict()
        data['id'] = doc.id
        result.append(data)
    
    return result

# delete event data
@router.delete("/{event_id}", response_model=dict)
def delete_event(event_id: str):
    doc_ref = event_collection.document(event_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Event data not found"
        )
    
    data = doc.to_dict()    
    files_mapping = {
        "post": "event_posts",
        "event_image": "event_images",
        "event_video": "event_videos"
    }

    deleted_status = delete_file_from_storage(data, files_mapping)   
    doc_ref.delete()

    return {
        "message": "Event delete successfully!",
        "event_id": event_id,
        "deleted_status": deleted_status
    }

# update event data
@router.put("/{event_id}", response_model=EventResponse)
def update_event(
    event_id: str,
    event_name: Optional[str] = Form(None),
    date: Optional[DateType] = Form(None),
    time: Optional[TimeType] = Form(None),
    venue: Optional[str] = Form(None),
    event_lat: Optional[float] = Form(None),
    event_lon: Optional[float] = Form(None),
    description: Optional[str] = Form(None),
    post: Optional[UploadFile] = File(None),
    event_image: List[UploadFile] = File(default=[]),
    remove_images: bool = Form(False),
    event_video: List[UploadFile] = File(default=[]),
    remove_video: bool = Form(False),
    email: Optional[EmailStr] = Form(None)
):
    doc_ref = event_collection.document(event_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Event not found")

    data = doc.to_dict()

    if event_name and date and time and venue:
        existing_event = (
            event_collection
            .where("event_name", "==", event_name)
            .where("venue", "==", venue)
            .where("date", "==", str(date))
            .where("time", "==", str(time))
            .get()
        )
        
        if any(e.id != event_id for e in existing_event):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="An event with the same name, date, time, and venue already exists."
            )
        
    event_image = [img for img in event_image if img.filename]
    event_video = [vid for vid in event_video if vid.filename]
    auto_remove_images = remove_images or (event_image == [] and event_image is not None)
    auto_remove_video = remove_video or (event_video == [] and event_video is not None)

    # Update files (replace or delete)
    data = update_file_in_storage(
        current_data=data,
        new_post=post,
        new_images=event_image if event_image else None,
        remove_images=auto_remove_images,
        new_video=event_video if event_video else None,
        remove_video=auto_remove_video
    )

    for field, value in {
        "event_name": event_name,
        "date": date,
        "time": time,
        "venue": venue,
        "event_lat": event_lat,
        "event_lon": event_lon,
        "description": description,
        "email": email
    }.items():
        if value is not None:
            data[field] = value

    update_dict = data.copy()
    if isinstance(update_dict.get("date"), DateType):
        update_dict["date"] = update_dict["date"].isoformat()
    if isinstance(update_dict.get("time"), TimeType):
        update_dict["time"] = update_dict["time"].strftime("%H:%M:%S")

    doc_ref.update(update_dict)

    return EventResponse(**data, id=event_id)

