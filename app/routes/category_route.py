from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from firebase_admin import storage
import uuid
from app.database.connection import db
from app.models.category import Category, CategoryOut
from typing import Optional
from urllib.parse import urlparse

# create router for category routes
router = APIRouter()
collection = db.collection('category') # firestore collection name

# add category
@router.post("/", response_model=CategoryOut)
def create_category(    
    category_id: str = Form(...),
    category_name: str = Form(...),
    category_type: str = Form(...),
    category_image: UploadFile = File(...)
):
    # Check if category name already exists
    existing_category = collection.where('category_name', '==', category_name).get()
    if existing_category:
        raise HTTPException(status_code=400, detail="This Category name already exists...!")
    
    # check if category_id already exists
    existing_category_id = collection.document(category_id).get()
    if existing_category_id.exists:
        raise HTTPException(status_code=400, detail="This Category ID already exists...!")  
    
    # upload image to firebase storage
    bucket = storage.bucket()
    image_id = str(uuid.uuid4())
    blob = bucket.blob(f'category_images/{image_id}_{category_image.filename}')
    blob.upload_from_file(category_image.file, content_type=category_image.content_type)
    blob.make_public()

    image_url = blob.public_url
        
    # Add new category 
    category_data = {
        "category_name": category_name,
        "category_type": category_type,
        "category_image": image_url
    }
    
    doc_ref = collection.document(category_id)
    doc_ref.set(category_data)
    return {"message": "Category added successfully", "category_id": category_id, **category_data}

# delete category
@router.delete("/{category_id}", response_model=dict)
def delete_category(category_id: str):
    doc_ref = collection.document(category_id)
    if not doc_ref.get().exists:
        raise HTTPException(status_code=404, detail="Category is not found")
    
    # Delete category image from firebase storage if it exists
    category_data = doc_ref.get().to_dict()
    image_delete = False

    if "category_image" in category_data:
        image_url = category_data["category_image"]
        parsed_url = urlparse(image_url)
        image_path = parsed_url.path.lstrip('/')
        bucket = storage.bucket()
        blob = bucket.blob(image_path)

        if blob.exists():
            blob.delete()
            image_delete = True
    
    doc_ref.delete()
    return {
        "message": "Category is deleted successfully",
        "category_id": category_id,
        "image_status": "Image deleted from Firebase Storage" if image_delete else "No image found or already deleted"
    }

# update category
@router.put("/{category_id}", response_model=CategoryOut)
def update_category(
    category_id: str,
    category_name: str = Form(...),
    category_type: str = Form(...),
    category_image: Optional[UploadFile] = File(None)
    ):

    doc_ref = collection.document(category_id)
    current_category = doc_ref.get()

    if not current_category.exists:
        raise HTTPException(status_code=404, detail="Category is not found")
    
    # Check if new category name already exists
    existing_category = collection.where('category_name', '==', category_name).get()
    if existing_category and existing_category[0].id != category_id:
        raise HTTPException(status_code=400, detail="This Category already exists...!")
    
    # Update category data
    category_data = {
        "category_name": category_name,
        "category_type": category_type
    }

    # If a new image is provided, delete old one and upload new one
    if category_image:

        # delete current image
        old_image = current_category.to_dict()
        old_image_url = old_image.get("category_image")
        if old_image_url:
            parsed_url = urlparse(old_image_url)
            image_path = parsed_url.path.lstrip('/')
            bucket = storage.bucket()
            blob = bucket.blob(image_path)

            if blob.exists():
                blob.delete()
                
        # upload new image 
        bucket = storage.bucket()
        image_id = str(uuid.uuid4())
        blob = bucket.blob(f'category_images/{image_id}_{category_image.filename}')
        blob.upload_from_file(category_image.file, content_type=category_image.content_type)
        blob.make_public()
        category_data["category_image"] = blob.public_url

    
    doc_ref.update(category_data)
    
    return {"message": "Category updated successfully", "category_id": category_id, **category_data}

# get all categories
@router.get("/", response_model=list[CategoryOut])
def get_all_categories():
    docs = collection.stream()
    categories = []

    for doc in docs:
        category_data = doc.to_dict()
        category_data['category_id'] = doc.id
        categories.append(category_data)
    
    return categories

# get category by id
@router.get("/{category_id}", response_model=CategoryOut)
def get_category_by_id(category_id: str):
    doc_ref = collection.document(category_id)
    category = doc_ref.get()

    if not category.exists:
        raise HTTPException(status_code=404, detail="Category is not found")
    
    category_data = category.to_dict()
    category_data['category_id'] = category.id
    return category_data

# get categories by type
@router.get("/type/{category_type}", response_model=list[CategoryOut])
def get_categories_by_type(category_type: str):
    docs = collection.where('category_type', '==', category_type).stream()
    categories = []

    for doc in docs:
        category_data = doc.to_dict()
        category_data['category_id'] = doc.id
        categories.append(category_data)
    
    if not categories:
        raise HTTPException(status_code=404, detail="No categories found for this type")
    
    return categories

