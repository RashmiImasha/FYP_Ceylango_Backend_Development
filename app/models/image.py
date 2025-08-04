from pydantic import BaseModel

class ImageDescriptionRequest(BaseModel):
    image_base64: str

class ImageDescriptionResponse(BaseModel):
    description: str
    
