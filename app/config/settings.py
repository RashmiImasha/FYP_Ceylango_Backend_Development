from pydantic import BaseSettings

class Settings(BaseSettings):
    FIREBASE_KEY_PATH: str
    FIREBASE_STORAGE_BUCKET: str

    class Config:
        env_file = ".env"

settings = Settings()