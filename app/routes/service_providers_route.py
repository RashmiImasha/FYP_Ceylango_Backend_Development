from typing import Optional
from typing_extensions import Literal
from fastapi import APIRouter, HTTPException
from firebase_admin import auth
from app.config.settings import settings
from app.models.user import ServiceProviderApplication
from app.database.connection import profiles_collection, user_collection
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import smtplib
import logging
from email.mime.text import MIMEText
from datetime import datetime
from urllib.parse import urlparse, parse_qs

# Extract oobCode from Firebase password reset link
def extract_oob_code(firebase_link: str) -> str:
    """Extract oobCode from Firebase password reset link"""
    parsed_url = urlparse(firebase_link)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('oobCode', [''])[0]

router = APIRouter()
security = HTTPBearer()
logger = logging.getLogger(__name__)


def send_email(to_email: str, subject: str, body: str) -> None:
    """Email sender function"""
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = settings.SMTP_EMAIL
    sender_password = settings.SMTP_PASS

    if not sender_email or not sender_password:
        logger.error("SMTP credentials not configured")
        raise Exception("[send_email] SMTP credentials not configured")

    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = sender_email
        msg["To"] = to_email

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, to_email, msg.as_string())

        logger.info(f"[send_email] Email successfully sent to {to_email} with subject '{subject}'")
    except Exception as e:
        logger.error(f"[send_email] Failed to send email to {to_email}: {str(e)}", exc_info=True)
        raise


@router.post("/apply")
async def apply_service_provider(user: ServiceProviderApplication):
    try:
        # Check if email already used in applications
        existing = user_collection.where("email", "==", user.email).stream()
        if any(existing):
            logger.info(f"Email already used in an application: {user.email}")
            raise HTTPException(status_code=400, detail="Email already used in an application")

        # Create user in Firestore
        doc_ref = user_collection.document()
        user_data = {
            "application_id": doc_ref.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": "service_provider",
            "disabled": False,
            "status": "Pending",
            "service_name": user.service_name,
            "district": user.district,
            "service_category": user.service_category,
            "phone_number": user.phone_number,
            "description": user.description,
            "applied_at": datetime.now().isoformat()
        }

        doc_ref.set(user_data)
        logger.info(f"Service provider application submitted: {user.email}, Application ID: {doc_ref.id}")
        return {
            "message": "Service provider application submitted successfully",
            "application_id": doc_ref.id,
            "status": "Pending",
            "service_category": user.service_category
        }

    except Exception as e:
        logger.error(f"Error submitting service provider application: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/review/{application_id}")
async def review_service_provider(application_id: str, decision: Literal["Approved", "Rejected"]):
    try:
        doc_ref = user_collection.document(application_id)
        user_doc = doc_ref.get()

        if not user_doc.exists:
            logger.warning(f"Service provider application not found: {application_id}")
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

            # Generate password reset link from Firebase
            reset_link = auth.generate_password_reset_link(user_data["email"])

            # Extract oobCode from Firebase link
            try:
                oob_code = extract_oob_code(reset_link)
                # Create frontend password setup URL
                frontend_reset_url = f" http://localhost:5173/setup-password?oobCode={oob_code}"
            except Exception as e:
                logger.error(f"Failed to extract oobCode: {str(e)}")
                raise HTTPException(status_code=500, detail="Failed to generate password reset link")

            # Create new document with uid as key
            new_doc_ref = user_collection.document(user_record.uid)
            user_data.update({
                "status": "Approved",
                "uid": user_record.uid,
                "role": "service_provider",
                "approved_at": datetime.now().isoformat()
            })
            new_doc_ref.set(user_data)

            # Create initial profile for service provider
            profile_data = {
                "uid": user_record.uid,
                "application_id": application_id,
                "service_name": user_data.get("service_name"),
                "service_category": user_data.get("service_category"),
                "description": user_data.get("description", ""),
                "district": user_data.get("district"),
                "phone_number": user_data.get("phone_number"),
                "address": "",
                "coordinates": None,
                "email": user_data.get("email"),
                "website": None,
                "social_media": None,
                "operating_hours": None,
                "profile_images": [],
                "poster_images": [],
                "amenities": [],
                "average_rating": 0.0,
                "total_reviews": 0,
                "rating_breakdown": {
                    "5": 0,
                    "4": 0,
                    "3": 0,
                    "2": 0,
                    "1": 0
                },
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            profiles_collection.document(user_record.uid).set(profile_data)

            # Delete old application document
            doc_ref.delete()

            # Send approval email with password setup link
            approval_email_body = f"""
Hi {user_data['full_name']},

Your application as a service provider has been approved! 

Click the link below to set your password and activate your account:

{frontend_reset_url}

This link expires in 24 hours.

If you did not apply for this, please contact our support team.

Regards,
Ceylango Team
            """

            send_email(
                to_email=user_data["email"],
                subject="Service Provider Application Approved",
                body=approval_email_body
            )

            logger.info(f"Service provider application approved: {user_data['email']}, UID: {user_record.uid}")
            return {
                "message": "Service provider approved and account created",
                "uid": user_record.uid,
                "service_category": user_data.get("service_category"),
                "service_id": application_id
            }

        else:  # Rejected
            doc_ref.delete()

            # Send rejection email
            rejection_email_body = f"""
Hi {user_data['full_name']},

Thank you for your interest in becoming a service provider.

Unfortunately, your service provider application has been rejected at this time.

If you believe this is a mistake or have any questions, please contact our support team.

Regards,
Salon Wave POS Team
            """

            send_email(
                to_email=user_data["email"],
                subject="Service Provider Application Rejected",
                body=rejection_email_body
            )

            logger.info(f"Service provider application rejected: {user_data['email']}, Application ID: {application_id}")
            return {"message": "Service provider application rejected"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reviewing service provider application {application_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/")
async def get_all_service_providers(
    status: Optional[str] = None,
    service_category: Optional[str] = None,
    district: Optional[str] = None
):
    """
    Fetch all service providers with optional filters
    """
    try:
        query = user_collection.where("role", "==", "service_provider")

        if status:
            query = query.where("status", "==", status)

        if service_category:
            query = query.where("service_category", "==", service_category)

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
                "service_category": data.get("service_category"),
                "phone_number": data.get("phone_number"),
                "status": data.get("status", "Pending"),
                "role": data.get("role"),
                "applied_at": data.get("applied_at"),
                "approved_at": data.get("approved_at")
            })

        logger.info(f"Fetched {len(providers)} service providers with filters - Status: {status}, Category: {service_category}, District: {district}")
        return {
            "count": len(providers),
            "service_providers": providers,
            "filters_applied": {
                "status": status,
                "service_category": service_category,
                "district": district
            }
        }

    except Exception as e:
        logger.error(f"Error fetching service providers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))