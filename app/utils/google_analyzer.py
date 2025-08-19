from app.config.settings import settings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
import base64
from enum import Enum

class ImageAnalysis(BaseModel):
    """
    Schema for analyzing an image.
    """
    location: str = Field(..., description="The location identified in the image")
    district: str = Field(..., description="The district of the location")
    description: str = Field(..., description="Historical and cultural value description")  

class CategoryEnum(str, Enum):
    Beach = "Beach"
    Waterfalls = "Waterfalls"
    Mountains = "Mountains"
    Historical = "Historical"
    Religious = "Religious"
    Adventure = "Adventure"
    Wildlife = "Wildlife"
    Others = "Others"

class DestinationAnalysis(BaseModel):
    destination_name: str = Field(..., description="Name of the destination")
    district_name: str = Field(..., description="District of the destination")
    raw_category_name: CategoryEnum = Field(..., description="Category of the destination")
    description: str = Field(..., description="Description of the destination")

def analyze_image_withAI(image_base64: str, prompt: str,api_label: str) -> ImageAnalysis:
    """
    Analyzes an image using the Gemini model and returns a structured output.
    
    Args:
        image_path: The file path to the image.
        prompt: The text prompt for the model.
        
    Returns:
        An instance of the ImageAnalysis Pydantic model.
    """
    api_key = settings.GOOGLE_API_KEY
    if not api_key:
        raise ValueError("Google API key is not set in the environment variables.")

    # Set up Gemini with LangChain
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=api_key,
    )

    # gemini-2.5-flash

    # 

    # Use .with_structured_output to enforce the Pydantic schema
    if api_label == "uploadImage":
        llm_with_structured_output = llm.with_structured_output(ImageAnalysis)
    else:
        llm_with_structured_output = llm.with_structured_output(DestinationAnalysis)
    
    
    # Construct the multimodal message
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    )
    response = llm_with_structured_output.invoke([message])
    
    print("Received structured response:")
    print(response)

    return response