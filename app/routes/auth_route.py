from fastapi import APIRouter, HTTPException, status, Depends
from firebase_admin import auth
from app.database.connection import db, user_collection
from app.models.user import UserCreate, UserLogin, UserInDB
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

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
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
@router.get("/login")
async def get_profile(current_user: dict = Depends(get_current_user)):
    try:
        # Fetch user document from Firestore using UID from Firebase Auth
        user_doc = user_collection.document(current_user['uid']).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        user_data = user_doc.to_dict()

        # Base response
        response = {
            "uid": user_data["uid"],
            "email": user_data["email"],
            "full_name": user_data.get("full_name"),
            "role": user_data.get("role", "tourist"),
            "disabled": user_data.get("disabled", False),
        }

        # Add optional service provider fields if they exist
        optional_fields = [
            "service_category",
            "service_name",
            "district",
            "status",
            "phone_number",
            "main_category",
            "sub_category",
        ]
        for field in optional_fields:
            if field in user_data:
                response[field] = user_data[field]

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching profile: {str(e)}")
