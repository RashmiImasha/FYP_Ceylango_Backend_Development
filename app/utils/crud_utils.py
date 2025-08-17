from fastapi import HTTPException
from app.utils.destinationUtils import delete_file_from_storage

def get_by_id(collection, doc_id: str):
    doc = collection.document(doc_id).get()
    if not doc.exists:
        raise HTTPException(status_code=404, detail="Document not found...!")
    
    data = doc.to_dict()
    data["id"] = doc.id
    return data

def get_all(collection):
    return [{**doc.to_dict(), "id": doc.id} for doc in collection.stream()]

def delete_by_id(collection, doc_id: str, files_mapping: dict = None):
    doc_ref = collection.document(doc_id)
    doc = doc_ref.get()

    if not doc.exists:
        raise HTTPException(status_code=404, detail="Document not found...!")
    
    data = doc.to_dict()

    deleted_files = {}
    if files_mapping:
        deleted_files = delete_file_from_storage(data, files_mapping)

    doc_ref.delete()
    return {
        "message": "Deleted successfully...!", 
        "id": doc_id, 
        "data": data,
        "deleted_images": deleted_files
    }

