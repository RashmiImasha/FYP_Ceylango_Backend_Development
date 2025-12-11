from fastapi import APIRouter, HTTPException, Depends
from firebase_admin import auth
from app.config import settings
from app.database.connection import user_collection
from app.models.user import ChangePasswordRequest,  UpdateUserProfileRequest, UserCreate, UserLogin, UserInDB
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio, logging
from datetime import datetime
import requests

logger = logging.getLogger(__name__)

router = APIRouter()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        if not credentials.credentials:
            logger.warning("No token provided in request")
            raise HTTPException(status_code=401, detail="No token provided")
        
        # Run the sync function in a thread pool
        loop = asyncio.get_event_loop()
        decoded_token = await loop.run_in_executor(
            None, 
            auth.verify_id_token,
            credentials.credentials
        )
        
        logger.info(f"Token verified for UID: {decoded_token['uid']}")
        return decoded_token
    except Exception as e:
        logger.error(f"Token verification failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")



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
        logger.info(f"User created with UID: {user_record.uid} role: {user_data['role']}")

        return {
            "message": "User created successfully",
            "uid": user_record.uid,
            "role": user_data["role"]
        }
    except firebase_exceptions.FirebaseError as e:
        logger.error(f"Firebase error during signup: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Firebase error: {str(e)}")
    except Exception as e:
        logger.error(f"Error during signup: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))



@router.get("/login")
async def get_profile(user_data: dict = Depends(get_current_user)):
    try:
        logger.info(f"Authenticated user UID: {user_data['uid']}")  # Debug log
        
        # Get additional user data from Firestore
        user_doc = user_collection.document(user_data['uid']).get()
        if not user_doc.exists:
            logger.error(f"User {user_data['uid']} not found in Firestore")  # Debug log
            raise HTTPException(status_code=404, detail="User not found")
        
        user_data_from_db = user_doc.to_dict()  # Renamed to avoid conflict
        logger.info(f"User data from Firestore: {user_data_from_db}")  # Debug log
        
        return {
            "uid": user_data_from_db["uid"],
            "email": user_data_from_db["email"],
            "full_name": user_data_from_db["full_name"],
            "role": user_data_from_db.get("role", "tourist")
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /login endpoint: {str(e)}")  # Debug log
        raise HTTPException(status_code=500, detail=str(e))
    



@router.put("/update-profile")
async def update_profile(
    profile_data: UpdateUserProfileRequest,
    user_data: dict = Depends(get_current_user)
):
    """
    Update user profile information (name only)
    """
    try:
        uid = user_data['uid']
        logger.info(f"Updating profile for UID: {uid}")
        
        # Update display name in Firebase Auth
        auth.update_user(uid, display_name=profile_data.full_name)
        
        # Update Firestore document
        user_collection.document(uid).update({
            "full_name": profile_data.full_name,
            "updated_at": datetime.now().isoformat()
        })
        
        logger.info(f"Profile updated successfully for UID: {uid}")
        return {
            "message": "Profile updated successfully",
            "full_name": profile_data.full_name
        }
    
    except Exception as e:
        logger.error(f"Error updating profile for UID {uid}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/change-password")
async def change_password(
    password_data: ChangePasswordRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Change user password - requires current password verification
    """
    try:
        # Verify the current token
        loop = asyncio.get_event_loop()
        decoded_token = await loop.run_in_executor(
            None,
            auth.verify_id_token,
            credentials.credentials
        )
        
        uid = decoded_token['uid']
        email = decoded_token.get('email')
        
        if not email:
            raise HTTPException(status_code=400, detail="User email not found")
        
        logger.info(f"Password change requested for UID: {uid}")
        
        # Verify current password by attempting to get a new token
        # This is done by making a request to Firebase's signInWithPassword endpoint
        import requests
        import json
        
        # Firebase API key - you should have this in your config
        firebase_api_key = settings.FIREBASE_API_KEY  # Add this to your settings
        
        # Verify current password
        verify_url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={firebase_api_key}"
        verify_payload = {
            "email": email,
            "password": password_data.current_password,
            "returnSecureToken": True
        }
        
        verify_response = requests.post(verify_url, json=verify_payload)
        
        if verify_response.status_code != 200:
            logger.warning(f"Current password verification failed for UID: {uid}")
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Validate new password length
        if len(password_data.new_password) < 6:
            raise HTTPException(status_code=400, detail="New password must be at least 6 characters")
        
        # Update password in Firebase Auth
        auth.update_user(uid, password=password_data.new_password)
        
        logger.info(f"Password changed successfully for UID: {uid}")
        return {
            "message": "Password changed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to change password")
