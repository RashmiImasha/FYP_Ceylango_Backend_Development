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

# collections
user_collection = db.collection("users")
category_collection = db.collection('category')
destination_collection = db.collection('destination')
emergancy_collection = db.collection("emergancyContact")
misplace_collection = db.collection('missingPlace')
event_collection = db.collection("events")
profiles_collection = db.collection("service_provider_profiles")
reviews_collection = db.collection("reviews")
chat_collection = db.collection("chat")
feedback_collection = db.collection("feedbacks")
