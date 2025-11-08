from fastapi import APIRouter, HTTPException, status, Depends
from firebase_admin import auth
from app.database.connection import db, user_collection
from app.models.user import UserCreate, UserLogin, UserInDB
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio

router = APIRouter()
security = HTTPBearer()

@router.post("/signup")
async def signup(user: UserCreate):
    try:
        # Create user in Firebase Auth
        user_record = auth.create_user(
            email=user.email,
            password=user.password,
            display_name=user.full_name
        )

        # Create user in Firestore
        user_data = {
            "uid": user_record.uid,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role if user.role else "tourist",
            "disabled": False,
        }

        user_collection.document(user_record.uid).set(user_data)

        return {
            "message": "User created successfully",
            "uid": user_record.uid,
            "role": user_data["role"]
        }
    except firebase_exceptions.FirebaseError as e:
        raise HTTPException(status_code=400, detail=f"Firebase error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        if not credentials.credentials:
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Run the sync function in a thread pool
        loop = asyncio.get_event_loop()
        decoded_token = await loop.run_in_executor(
            None, 
            auth.verify_id_token, 
            credentials.credentials
        )
        
        print(f"Token verified for UID: {decoded_token['uid']}")
        return decoded_token
    except Exception as e:
        print(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")


@router.get("/login")
async def get_profile(user_data: dict = Depends(get_current_user)):
    try:
        print(f"Authenticated user UID: {user_data['uid']}")  # Debug log
        
        # Get additional user data from Firestore
        user_doc = user_collection.document(user_data['uid']).get()
        if not user_doc.exists:
            print(f"User {user_data['uid']} not found in Firestore")  # Debug log
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data_from_db = user_doc.to_dict()  # Renamed to avoid conflict
        print(f"User data from Firestore: {user_data_from_db}")  # Debug log
        
        return {
            "uid": user_data_from_db["uid"],
            "email": user_data_from_db["email"],
            "full_name": user_data_from_db["full_name"],
            "role": user_data_from_db.get("role", "tourist")
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in /login endpoint: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))