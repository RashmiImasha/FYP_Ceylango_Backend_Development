from fastapi import UploadFile, HTTPException
from firebase_admin import storage
from app.database.connection import category_collection, destination_collection
from typing import Optional, List
from urllib.parse import urlparse, unquote
from PIL import Image
from datetime import datetime
import uuid, imagehash, math, logging

logger = logging.getLogger(__name__)

class CrudUtils:

    # ****************************************************
    #  Storage Handle Utilities
    # ****************************************************

    @staticmethod
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

    @staticmethod
    def delete_file_from_storage(data:dict, files_mapping: dict) -> dict:
        """
        Deletes files from Firebase Storage based on the data dictionary.
                
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
                    path = urlparse(url).path  
                    parts = path.split("/")
                    bucket_name = parts[1]
                    file_path = "/".join(path.split("/")[2:])  
                    file_path = unquote(file_path) 

                    bucket = storage.bucket(bucket_name)  # use correct bucket
                    blob = bucket.blob(file_path)
                    
                    if blob.exists():
                        blob.delete()
                        deleted.append(file_path)

            if deleted:
                logger.info(f"Deleted files from {field}: {deleted}")
            else:
                logger.warning(f"No files found to delete in {field}")

            deleted_status[field] = f"Deleted: {deleted}" if deleted else "No file found or already deleted"
        
        return deleted_status

    @staticmethod
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
            CrudUtils.delete_file_from_storage({"post": data.get("post")}, {"post": "event_posts"})
            post_url = CrudUtils.upload_file_to_storage(new_post, "event_posts")
            data["post"] = post_url
        
        # replace or remove image
        if new_images:
            CrudUtils.delete_file_from_storage({"event_image": data.get("event_image")}, {"event_image": "event_images"})
            img_urls = []
            for img in new_images:
                url = CrudUtils.upload_file_to_storage(img, "event_images")
                if url:
                    img_urls.append(url)
            data["event_image"] = img_urls
        elif remove_images:
            CrudUtils.delete_file_from_storage({"event_image": data.get("event_image")}, {"event_image": "event_images"})
            data["event_image"] = None

            
        # Replace or remove videos
        if new_video:
            CrudUtils.delete_file_from_storage({"event_video": data.get("event_video")}, {"event_video": "event_videos"})
            vid_urls = []
            for vid in new_video:
                url = CrudUtils.upload_file_to_storage(vid, "event_videos")
                if url:
                    vid_urls.append(url)
            data["event_video"] = vid_urls
        elif remove_video:
            CrudUtils.delete_file_from_storage({"event_video": data.get("event_video")}, {"event_video": "event_videos"})
            data["event_video"] = None
        
        return data
                        
    @staticmethod
    def move_files_to_new_folder(urls: list, source_folder: str, target_folder: str) -> list:
        bucket = storage.bucket()
        new_urls = []

        for url in urls:
            path = urlparse(url).path
            file_name = "/".join(path.split("/")[2:])  
            file_name = unquote(file_name)

            # Download the blob content
            blob = bucket.blob(file_name)
            if not blob.exists():
                continue

            # Create a new blob in the destination folder
            new_blob_name = f"{target_folder}/{file_name.split('/')[-1]}"
            new_blob = bucket.blob(new_blob_name)
            new_blob.rewrite(blob)  # copy content to new blob
            new_blob.make_public()

            new_urls.append(new_blob.public_url)
            print(f"Moved {file_name} → {new_blob_name}")

            # Delete original blob
            blob.delete()
            logger.info(f"Deleted original image: {file_name}")

        return new_urls


    # ****************************************************
    #  Destination Utils
    # ****************************************************

    @staticmethod
    def haversine(lat1, lon1, lat2, lon2):
        # calculate distance between two points on the Earth ( Radius )
        R = 6371000
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi/2.0)**2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2.0)**2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    @staticmethod
    def compute_phash(upload_file) -> str:
        upload_file.seek(0)
        image = Image.open(upload_file)
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if image.width < 32 or image.height < 32:
            image = image.resize((32, 32))
        
        phash = imagehash.phash(image, hash_size=8)
        return str(phash)

    @staticmethod
    def add_destination_record(destination_data: dict, images=None):
        
        destination_name = destination_data["destination_name"]
        latitude = destination_data["latitude"]
        longitude = destination_data["longitude"]
        district_name = destination_data["district_name"]
        description = destination_data["description"]
        category_name = destination_data["category_name"]

        # Check if category exists
        category_query = category_collection \
            .where('category_name', '==', category_name) \
            .where('category_type', '==', 'location') \
            .get()
        if not category_query:
            raise HTTPException(status_code=404, detail="Category not found or not of type 'location'")

        # Check duplicate destination name
        existing_destination = destination_collection.where('destination_name', '==', destination_name).get()
        if existing_destination:
            raise HTTPException(status_code=400, detail="This Destination name already exists...!")

        # Check nearby (within 5m)
        for doc in destination_collection.stream():
            data = doc.to_dict()
            existing_lat = data.get("latitude")
            existing_lon = data.get("longitude")
            if existing_lat is not None and existing_lon is not None:
                distance = CrudUtils.haversine(latitude, longitude, existing_lat, existing_lon)
                if distance < 5:
                    raise HTTPException(status_code=400, detail="A destination already exists very close to this location")

        image_urls = []
        phash_list = []

        if images:  # Case: create_destination with UploadFile
            allow_types = ['image/jpeg', 'image/jpg', 'image/png']
            for img in images:
                if img.content_type not in allow_types:
                    raise HTTPException(status_code=400, detail="Unsupported image file type")

            existing_phash_dicts = [doc.to_dict().get('image_phash', []) for doc in destination_collection.stream()]

            for img in images:
                img.file.seek(0)
                new_phash = CrudUtils.compute_phash(img.file)

                for existing_phash_list in existing_phash_dicts:
                    for existing_phash in existing_phash_list:
                        distance = imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(new_phash)
                        if distance <= 5:
                            raise HTTPException(status_code=400, detail="A visually similar image already exists.")

                url = CrudUtils.upload_file_to_storage(img, "destination_images")
                if url:
                    image_urls.append(url)
                    phash_list.append(new_phash)

        else:  # Case: move from missingplace (already has image + phash)
            image_urls = destination_data["destination_image"]
            phash_list = destination_data["image_phash"]

        # Save record
        record = {
            "destination_name": destination_name.strip(),
            "latitude": latitude,
            "longitude": longitude,
            "district_name": district_name.strip(),
            "district_name_lower": district_name.strip().lower(),
            "description": description,
            "destination_image": image_urls,
            "image_phash": phash_list,
            "category_name": category_name.strip(),
            "average_rating": 0.0,
            "total_reviews": 0,
            "rating_breakdown": {
                "1": 0,
                "2": 0, 
                "3": 0,
                "4": 0,
                "5": 0
            },
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
            
        }

        _, doc_ref = destination_collection.add(record)
        record["id"] = doc_ref.id
        return record

    @staticmethod
    def update_destination_record(
        doc_ref,
        destination_id: str,
        collection,  
        destination_name: str,
        latitude: float,
        longitude: float,
        district_name: str,
        description: str,
        category_name: str,
        new_images: Optional[List[UploadFile]] = None,
        remove_existing: Optional[List[str]] = None
    ):
        """
        Shared update function for both destination and missingplace.
        """

        destination = doc_ref.get()
        if not destination.exists:
            raise HTTPException(status_code=404, detail="Record not found")

        current_data = destination.to_dict()
        current_data["id"] = destination_id

        # validate category
        category_query = category_collection \
            .where('category_name', '==', category_name) \
            .where('category_type', '==', 'location') \
            .get()
        if not category_query:
            raise HTTPException(status_code=404, detail="Valid location category not found")

        # duplicate name check
        existing_destination = collection.where('destination_name', '==', destination_name).stream()
        for doc in existing_destination:
            if doc.id != destination_id:
                raise HTTPException(status_code=400, detail="This Destination name already exists...!")

        # nearby check
        for doc in collection.stream():
            if doc.id == destination_id:
                continue
            data = doc.to_dict()
            if data.get("latitude") is not None and data.get("longitude") is not None:
                distance = CrudUtils.haversine(latitude, longitude, data["latitude"], data["longitude"])
                if distance < 5:
                    raise HTTPException(status_code=400, detail="Another destination is too close to this location")

        # handle images
        current_images = current_data.get("destination_image", []) or []
        current_phash = current_data.get("image_phash", []) or []

        if remove_existing:
            for url in remove_existing:
                if url in current_images:
                    idx = current_images.index(url)
                    current_images.pop(idx)
                    current_phash.pop(idx)
                    CrudUtils.delete_file_from_storage({"destination_image": [url]}, {"destination_image": "destination_images"})

        if new_images:
            allowed_types = ["image/jpeg", "image/jpg", "image/png"]
            existing_phash_lists = [
                doc.to_dict().get("image_phash", []) for doc in collection.stream() if doc.id != destination_id
            ]
            current_phash_set = set(current_phash)

            for img in new_images:
                if img.content_type not in allowed_types:
                    raise HTTPException(status_code=400, detail="Unsupported image file type")
                img.file.seek(0)
                phash = CrudUtils.compute_phash(img.file)

                if phash in current_phash_set:
                    raise HTTPException(status_code=400, detail="This image already exists in this Image List...!")

                for phash_list in existing_phash_lists:
                    for existing_phash in phash_list:
                        if imagehash.hex_to_hash(existing_phash) - imagehash.hex_to_hash(phash) <= 5:
                            raise HTTPException(status_code=400, detail="A visually similar image already exists.")

                url = CrudUtils.upload_file_to_storage(img, "destination_images")
                if url:
                    current_images.append(url)
                    current_phash.append(phash)
                    current_phash_set.add(phash)

        if len(current_images) == 0:
            raise HTTPException(status_code=400, detail="At least one image must remain")

        # update data
        current_data.update({
            "destination_name": destination_name.strip(),
            "latitude": latitude,
            "longitude": longitude,
            "district_name": district_name.strip(),
            "district_name_lower": district_name.strip().lower(),
            "description": description,
            "category_name": category_name.strip(),
            "destination_image": current_images,
            "image_phash": current_phash,
            "updated_at": datetime.now().isoformat()
        })

        doc_ref.set(current_data, merge=False)
        return {"id": destination_id, **current_data}


    # ****************************************************
    #  CRUD Utils
    # ****************************************************

    @staticmethod
    def get_by_id(collection, doc_id: str):
        doc = collection.document(doc_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found...!")
        
        data = doc.to_dict()
        data["id"] = doc.id
        return data

    @staticmethod
    def get_all(collection):
        return [{**doc.to_dict(), "id": doc.id} for doc in collection.stream()]

    @staticmethod
    def delete_by_id(collection, doc_id: str, files_mapping: dict = None):
        doc_ref = collection.document(doc_id)
        doc = doc_ref.get()

        if not doc.exists:
            raise HTTPException(status_code=404, detail="Document not found...!")
        
        data = doc.to_dict()

        deleted_files = {}
        if files_mapping:
            deleted_files = CrudUtils.delete_file_from_storage(data, files_mapping)

        doc_ref.delete()
        logger.info(f"Deleted document ID: {doc_id} from collection.")
        return {
            "message": "Deleted successfully...!", 
            "id": doc_id, 
            "data": data,
            "deleted_images": deleted_files
        }
