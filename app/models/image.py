from pydantic import BaseModel, Field

class ImageDescriptionRequest(BaseModel):
    image_base64: str

class ImageDescriptionResponse(BaseModel):
    description: str
    
class ImageAnalysis(BaseModel):
    """
    Schema for analyzing an image.
    """
    location: str = Field(..., description="The location identified in the image")
    district: str = Field(..., description="The district of the location")
    description: str = Field(..., description="Historical and cultural value description")
    
