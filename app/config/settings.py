from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    FIREBASE_KEY_PATH: str
    FIREBASE_STORAGE_BUCKET: str
    FIREBASE_API_KEY: str
    GOOGLE_API_KEY: str
    
    SMTP_EMAIL: str
    SMTP_PASS: str

    class Config:
        env_file = ".env"

settings = Settings()
