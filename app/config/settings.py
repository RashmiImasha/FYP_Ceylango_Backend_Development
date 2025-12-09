from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    FIREBASE_KEY_PATH: str
    FIREBASE_STORAGE_BUCKET: str
    FIREBASE_API_KEY: str
    GOOGLE_API_KEY: str
    
    SMTP_EMAIL: str
    SMTP_PASS: str

    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: Optional[str] = "us-east-1"
    PINECONE_INDEX_NAME: str

    TEXT_EMBEDDING_MODEL: str
    CLIP_MODEL: str
    GEMINI_MODEL: str
    OSRM_URL: str

    class Config:
        env_file = ".env"

settings = Settings()
