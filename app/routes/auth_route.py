from fastapi import APIRouter, HTTPException, status
from firebase_admin import auth
from app.database.connection import db
from app.models.user import UserCreate, UserLogin
from firebase_admin import exceptions as firebase_exceptions
from fastapi import APIRouter, Depends
from app.routes.firebase_auth import verify_token

router = APIRouter()
collection = db.collection("users")

@router.post("/signup")
def signup(user: UserCreate):
    try:
        user_record = auth.create_user(
            email=user.email,
            password=user.password,
            display_name=user.full_name
        )

        user_data = {
            "uid": user_record.uid,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role if user.role else "tourist",
            "disabled": False,
        }

        collection.document(user_record.uid).set(user_data)

        return {
            "message": "User created successfully",
            "uid": user_record.uid,
            "role": user_data["role"]
        }

    except firebase_exceptions.FirebaseError as e:
        raise HTTPException(status_code=400, detail=f"Firebase error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/login")
def login(user: UserLogin):
    
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Login should be handled via Firebase Client SDK (React/Flutter). Send ID token to backend for protected access.",
    )

@router.get("/me")
def get_profile(user_data=Depends(verify_token)):
    
    return {
        "uid": user_data["uid"],
        "email": user_data["email"],
        "role": user_data.get("role", "tourist")
    }
    
