import uuid
from firebase_admin import storage
from fastapi import UploadFile
from typing import Optional, List

def upload_file_to_storage(file: Optional[UploadFile], folder:str) -> Optional[str]:
    """
    Uploads a file to Firebase Storage and returns the public URL.
    Returns None if no file is provided.

    """

    if not file:
        return None
    
    bucket = storage.bucket()
    file_id = str(uuid.uuid4())
    blob = bucket.blob(f"{folder}/{file_id}_{file.filename}")
    file.file.seek(0)
    blob.upload_from_file(file.file)
    blob.make_public()
    return blob.public_url

def delete_file_from_storage(data:dict, files_mapping: dict) -> dict:
    """
    Deletes files from Firebase Storage based on the data dictionary.
    
    Args:
        data (dict): Firestore document data containing URLs.
        files_mapping (dict): Mapping of Firestore fields to Storage folders.
    
    Returns:
        dict: Status of deleted files for each field.
    """
    bucket = storage.bucket()
    deleted_status = {}
    
    for field, folder in files_mapping.items():
        deleted = []
        if field in data and data[field]:
            urls = data[field] if isinstance(data[field], list) else [data[field]]
            for url in urls:
                file_name = url.split('/')[-1]
                blob = bucket.blob(f"{folder}/{file_name}")
                if blob.exists():
                    blob.delete()
                    deleted.append(file_name)

        deleted_status[field] = f"Deleted: {deleted}" if deleted else "No file found or already deleted"
    
    return deleted_status

def update_file_in_storage(
        current_data: dict,
        new_post: Optional[UploadFile] = None,
        new_images: Optional[List[UploadFile]] = None,
        remove_images: bool = False,
        new_video: Optional[List[UploadFile]] = None,
        remove_video: bool = False
) -> dict:
    """
    Handles updating post, images, and videos:
    - Deletes old files if replaced or explicitly removed.
    - Uploads new files if provided.
    """
    data = current_data.copy()

    # Replace post if new one is provided
    if new_post:
        delete_file_from_storage({"post": data.get("post")}, {"post": "event_posts"})
        post_url = upload_file_to_storage(new_post, "event_posts")
        data["post"] = post_url
    
    # replace or remove image
    if new_images:
        delete_file_from_storage({"event_image": data.get("event_image")}, {"event_image": "event_images"})
        img_urls = []
        for img in new_images:
            url = upload_file_to_storage(img, "event_images")
            if url:
                img_urls.append(url)
        data["event_image"] = img_urls
    elif remove_images:
        delete_file_from_storage({"event_image": data.get("event_image")}, {"event_image": "event_images"})
        data["event_image"] = None

        
    # Replace or remove videos
    if new_video:
        delete_file_from_storage({"event_video": data.get("event_video")}, {"event_video": "event_videos"})
        vid_urls = []
        for vid in new_video:
            url = upload_file_to_storage(vid, "event_videos")
            if url:
                vid_urls.append(url)
        data["event_video"] = vid_urls
    elif remove_video:
        delete_file_from_storage({"event_video": data.get("event_video")}, {"event_video": "event_videos"})
        data["event_video"] = None
    
    return data


        
                       

