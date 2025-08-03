import firebase_admin
from firebase_admin import credentials, firestore, storage   
from app.config.settings import settings

# Initialize Firebase App
if not firebase_admin._apps:
    cred = credentials.Certificate(settings.FIREBASE_KEY_PATH)
    firebase_admin.initialize_app(cred, {
    'storageBucket': settings.FIREBASE_STORAGE_BUCKET
    })

# Initialize Firestore DB
db = firestore.client()


