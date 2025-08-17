from typing import Optional
from typing_extensions import Literal
from fastapi import APIRouter, HTTPException, status, Depends
from firebase_admin import auth
from app.database.connection import db
from app.models.user import ServiceProviderApplication,  UserCreate, UserLogin, UserInDB
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import smtplib
from email.mime.text import MIMEText

router = APIRouter()
collection = db.collection("users")
security = HTTPBearer()

   
@router.post("/apply")
async def apply_service_provider(user: ServiceProviderApplication):
    try:
        user_data = user.dict()
        user_data["role"] = "service_provider"
        user_data["status"] = "Pending"

        doc_ref = collection.document()
        user_data["application_id"] = doc_ref.id
        doc_ref.set(user_data)

        return {
            "message": "Service provider application submitted successfully",
            "application_id": doc_ref.id,
            "status": "Pending"
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
            # Generate random password
            generated_password = secrets.token_urlsafe(8)  # e.g., "aB3kd82L"
            
            # Create user in Firebase Auth
            user_record = auth.create_user(
                email=user_data["email"],
                password=generated_password,
                display_name=user_data["full_name"]
            )

            # Update Firestore with approved status + UID
            doc_ref.update({
                "status": "Approved",
                "uid": user_record.uid,
            })

            # Send email with login credentials
            send_email(
                to_email=user_data["email"],
                subject="Service Provider Application Approved",
                body=f"""
                Hi {user_data['full_name']},

                Your application as a service provider has been approved. 
                You can now log in to your dashboard.

                Email: {user_data['email']}
                Password: {generated_password}

                Please change your password after login.

                Regards,
                Tourist App Team
                """
            )

            return {"message": "Service provider approved and account created", "uid": user_record.uid}

        else:  # Rejected
            doc_ref.update({"status": "Rejected"})

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


def send_email(to_email: str, subject: str, body: str):
    """ Example SMTP Email Sender (you can replace with SendGrid, AWS SES, etc.) """
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender_email = "subanyakalpani46@gmail.com"
    sender_password = "yzei nzag osrk rdvb"

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender_email
    msg["To"] = to_email

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, to_email, msg.as_string())

@router.get("/")
async def get_all_service_providers(status: Optional[str] = None):
    """
    Fetch all service providers.
    Optional query param ?status=Approved or ?status=Pending
    """
    try:
        query = collection.where("role", "==", "service_provider")
        
        # If status filter provided
        if status:
            query = query.where("status", "==", status)

        docs = query.stream()
        providers = []

        for doc in docs:
            data = doc.to_dict()
            providers.append({
                "application_id": data.get("application_id", doc.id),
                "uid": data.get("uid"),
                "email": data.get("email"),
                "full_name": data.get("full_name"),
                "service_name": data.get("service_name"),
                "district": data.get("district"),
                "service_category": data.get("service_category"),
                "phone_number": data.get("phone_number"),
                "status": data.get("status", "Pending"),
            })

        return {"count": len(providers), "service_providers": providers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
