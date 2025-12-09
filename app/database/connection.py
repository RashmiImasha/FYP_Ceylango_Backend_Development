import firebase_admin, os, json
from firebase_admin import credentials, firestore   
from app.config.settings import settings

# Initialize Firebase App
if not firebase_admin._apps:

    firebase_json = os.getenv("FIREBASE_JSON")
    if firebase_json:
        # Running on RAILWAY 
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)

    else:
        # Running LOCALLY → load from file
        cred = credentials.Certificate(settings.FIREBASE_KEY_PATH)

    firebase_admin.initialize_app(cred, {
        "storageBucket": settings.FIREBASE_STORAGE_BUCKET
    })

    # cred = credentials.Certificate(settings.FIREBASE_KEY_PATH)
    # firebase_admin.initialize_app(cred, {
    # 'storageBucket': settings.FIREBASE_STORAGE_BUCKET
    # })

# Initialize Firestore DB
db = firestore.client()

# collections
user_collection = db.collection("users")
category_collection = db.collection('category')
destination_collection = db.collection('destination')
emergancy_collection = db.collection("emergancyContact")
misplace_collection = db.collection('missingPlace')
profiles_collection = db.collection("service_provider_profiles")
reviews_collection = db.collection("reviews")
chat_collection = db.collection("chat")
feedback_collection = db.collection("feedbacks")
chatbot_history_collection = db.collection("chatbot_history")
tripPlan_collection = db.collection("trip_plans")