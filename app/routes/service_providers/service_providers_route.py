from typing import Optional, Dict, Any
from typing_extensions import Literal
from fastapi import APIRouter, HTTPException, status, Depends
from firebase_admin import auth
from app.config.settings import settings
from app.database.connection import db
from app.models.user import (
    ServiceProviderApplication, ServiceProviderProfile, BaseServiceProfile,
    MainCategory, AccommodationProfile, FoodDiningProfile, WellnessProfile,
    ShoppingProfile, ActivitiesProfile, TransportationProfile,
    AccommodationSubCategory, FoodDiningSubCategory, WellnessSubCategory,
    ShoppingSubCategory, ActivitiesSubCategory, TransportationSubCategory
)
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

router = APIRouter()
collection = db.collection("users")
profiles_collection = db.collection("service_provider_profiles")
security = HTTPBearer()

# Category validation mapping
CATEGORY_MAPPING = {
    MainCategory.ACCOMMODATION: [e.value for e in AccommodationSubCategory],
    MainCategory.FOOD_DINING: [e.value for e in FoodDiningSubCategory],
    MainCategory.WELLNESS: [e.value for e in WellnessSubCategory],
    MainCategory.SHOPPING: [e.value for e in ShoppingSubCategory],
    MainCategory.ACTIVITIES: [e.value for e in ActivitiesSubCategory],
    MainCategory.TRANSPORTATION: [e.value for e in TransportationSubCategory],
}

def validate_category_combination(main_category: MainCategory, sub_category: str):
    """Validate if sub_category belongs to main_category"""
    valid_subcategories = CATEGORY_MAPPING.get(main_category, [])
    if sub_category not in valid_subcategories:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid subcategory '{sub_category}' for main category '{main_category}'"
        )

@router.post("/apply")
async def apply_service_provider(user: ServiceProviderApplication):
    try:
        # Validate category combination
        validate_category_combination(user.main_category, user.sub_category)
        
        # Check if email already used in applications
        existing = collection.where("email", "==", user.email).stream()
        if any(existing):
            raise HTTPException(status_code=400, detail="Email already used in an application")

        # Create user in Firestore
        doc_ref = collection.document()
        user_data = {
            "application_id": doc_ref.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": "service_provider",
            "disabled": False,
            "status": "Pending",
            "service_name": user.service_name,
            "district": user.district,
            "main_category": user.main_category,
            "sub_category": user.sub_category,
            "phone_number": user.phone_number,
            "description": user.description,
            "applied_at": datetime.now().isoformat()
        }

        doc_ref.set(user_data)

        return {
            "message": "Service provider application submitted successfully",
            "application_id": doc_ref.id,
            "status": "Pending",
            "category_info": {
                "main_category": user.main_category,
                "sub_category": user.sub_category
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/review/{application_id}")
async def review_service_provider(application_id: str, decision: Literal["Approved", "Rejected"]):
    try:
        doc_ref = collection.document(application_id)
        user_doc = doc_ref.get()

        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="Application not found")

        user_data = user_doc.to_dict()

        if decision == "Approved":
            # Create user in Firebase Auth (if not already exists)
            try:
                user_record = auth.get_user_by_email(user_data["email"])
            except firebase_exceptions.NotFoundError:
                user_record = auth.create_user(
                    email=user_data["email"],
                    display_name=user_data["full_name"]
                )

            # Generate password reset link
            reset_link = auth.generate_password_reset_link(user_data["email"])

            new_doc_ref = collection.document(user_record.uid)
            user_data.update({
                "status": "Approved",
                "uid": user_record.uid,
                "role": "service_provider",
                "approved_at": datetime.now().isoformat()
            })
            new_doc_ref.set(user_data)

            # Create empty profile document for the service provider
            await create_empty_profile(user_record.uid, user_data)

            # Delete old document
            doc_ref.delete()

            # Send approval email
            send_email(
                to_email=user_data["email"],
                subject="Service Provider Application Approved",
                body=f"""
                Hi {user_data['full_name']},

                Your application as a service provider has been approved. 
                Please set your password and login using the link below:

                {reset_link}

                You can now access your service provider dashboard and complete your profile.

                Regards,
                Tourist App Team
                """
            )

            return {
                "message": "Service provider approved and account created", 
                "uid": user_record.uid,
                "category_info": {
                    "main_category": user_data.get("main_category"),
                    "sub_category": user_data.get("sub_category")
                }
            }

        else:  # Rejected
            doc_ref.delete()

            # Send rejection email
            send_email(
                to_email=user_data["email"],
                subject="Service Provider Application Rejected",
                body=f"""
                Hi {user_data['full_name']},

                Unfortunately, your service provider application has been rejected. 
                If you think this is a mistake, please contact support.

                Regards,
                Tourist App Team
                """
            )

            return {"message": "Service provider application rejected"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def create_empty_profile(uid: str, user_data: Dict[str, Any]):
    """Create an empty profile document for approved service provider"""
    main_category = user_data.get("main_category")
    sub_category = user_data.get("sub_category")
    
    # Create base profile structure
    profile_data = {
        "uid": uid,
        "main_category": main_category,
        "sub_category": sub_category,
        "profile_completed": False,
        "base_info": {
            "service_name": user_data.get("service_name", ""),
            "description": user_data.get("description", ""),
            "address": "",
            "district": user_data.get("district", ""),
            "phone_number": user_data.get("phone_number", ""),
            "operating_hours": {},
            "images": [],
            "amenities": [],
            "is_active": True
        },
        "category_data": get_empty_category_data(main_category),
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    profiles_collection.document(uid).set(profile_data)

def get_empty_category_data(main_category: MainCategory) -> Dict[str, Any]:
    """Return empty category-specific data structure"""
    if main_category == MainCategory.ACCOMMODATION:
        return {
            "room_types": [],
            "check_in_time": "14:00",
            "check_out_time": "11:00",
            "room_amenities": [],
            "hotel_amenities": [],
            "price_range": {}
        }
    elif main_category == MainCategory.FOOD_DINING:
        return {
            "cuisine_types": [],
            "menu_items": [],
            "dietary_options": [],
            "average_meal_price": {},
            "delivery_available": False,
            "takeaway_available": False
        }
    elif main_category == MainCategory.WELLNESS:
        return {
            "services_offered": [],
            "therapists": [],
            "treatment_packages": [],
            "booking_advance_days": 7,
            "cancellation_hours": 24
        }
    elif main_category == MainCategory.SHOPPING:
        return {
            "product_categories": [],
            "inventory_items": [],
            "payment_methods": [],
            "shipping_available": False
        }
    elif main_category == MainCategory.ACTIVITIES:
        return {
            "activity_types": [],
            "group_size": {},
            "equipment_provided": [],
            "price_per_person": None
        }
    elif main_category == MainCategory.TRANSPORTATION:
        return {
            "vehicle_types": [],
            "coverage_areas": [],
            "booking_advance_hours": 2,
            "driver_available": True,
            "insurance_included": True
        }
    else:
        return {}

@router.get("/")
async def get_all_service_providers(
    status: Optional[str] = None,
    main_category: Optional[MainCategory] = None,
    district: Optional[str] = None
):
    """
    Fetch all service providers with optional filters
    """
    try:
        query = collection.where("role", "==", "service_provider")

        if status:
            query = query.where("status", "==", status)
        
        if main_category:
            query = query.where("main_category", "==", main_category)
            
        if district:
            query = query.where("district", "==", district)

        docs = query.stream()
        providers = []

        for doc in docs:
            data = doc.to_dict()
            providers.append({
                "application_id": data.get("application_id"),
                "uid": data.get("uid"),
                "email": data.get("email"),
                "full_name": data.get("full_name"),
                "service_name": data.get("service_name"),
                "district": data.get("district"),
                "main_category": data.get("main_category"),
                "sub_category": data.get("sub_category"),
                "phone_number": data.get("phone_number"),
                "status": data.get("status", "Pending"),
                "role": data.get("role"),
                "applied_at": data.get("applied_at"),
                "approved_at": data.get("approved_at")
            })

        return {
            "count": len(providers), 
            "service_providers": providers,
            "filters_applied": {
                "status": status,
                "main_category": main_category,
                "district": district
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/categories")
async def get_categories():
    """Get all available categories and subcategories"""
    return {
        "categories": {
            category.value: {
                "name": category.value.replace("_", " ").title(),
                "subcategories": [
                    {
                        "value": sub,
                        "name": sub.replace("_", " ").title()
                    } for sub in subcategories
                ]
            }
            for category, subcategories in CATEGORY_MAPPING.items()
        }
    }

def send_email(to_email: str, subject: str, body: str):
    """Email sender function"""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = settings.SMTP_EMAIL
    sender_password = settings.SMTP_PASS

    if not sender_email or not sender_password:
        raise Exception("SMTP credentials not configured")

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())
    except Exception as e:
        print(f"❌ Email sending failed: {e}")  # log error but don't block approval