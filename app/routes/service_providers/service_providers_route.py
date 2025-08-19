from typing import Optional
from typing_extensions import Literal
from fastapi import APIRouter, HTTPException, status, Depends
from firebase_admin import auth
from app.config.settings import settings
from app.database.connection import db
from app.models.user import ServiceProviderApplication
from firebase_admin import exceptions as firebase_exceptions
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import smtplib
from email.mime.text import MIMEText


router = APIRouter()
collection = db.collection("users")
security = HTTPBearer()


@router.post("/apply")
async def apply_service_provider(user: ServiceProviderApplication):
    try:
        # ✅ Check if email already used in applications
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
            "service_category": user.service_category,
            "phone_number": user.phone_number
        }

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
            # ✅ Create user in Firebase Auth (if not already exists)
            try:
                user_record = auth.get_user_by_email(user_data["email"])
            except firebase_exceptions.NotFoundError:
                user_record = auth.create_user(
                    email=user_data["email"],
                    display_name=user_data["full_name"]
                )

            # ✅ Generate password reset link instead of sending plain password
            reset_link = auth.generate_password_reset_link(user_data["email"])

            new_doc_ref = collection.document(user_record.uid)
            user_data.update({
                "status": "Approved",
                "uid": user_record.uid,
                "role": "service_provider"
            })
            new_doc_ref.set(user_data)

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

                Regards,
                Tourist App Team
                """
            )

            return {"message": "Service provider approved and account created", "uid": user_record.uid}

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



def send_email(to_email: str, subject: str, body: str):
    """ Safer SMTP Email Sender using env vars """
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


@router.get("/")
async def get_all_service_providers(status: Optional[str] = None):
    """
    Fetch all service providers.
    Optional query param ?status=Approved or ?status=Pending
    """
    try:
        query = collection.where("role", "in", ["service_provider", "pending_service_provider"])

        if status:
            query = query.where("status", "==", status)

        docs = query.stream()
        providers = []

        for doc in docs:
            data = doc.to_dict()
            providers.append({
                "application_id": doc.id,
                "uid": data.get("uid"),
                "email": data.get("email"),
                "full_name": data.get("full_name"),
                "service_name": data.get("service_name"),
                "district": data.get("district"),
                "service_category": data.get("service_category"),
                "phone_number": data.get("phone_number"),
                "status": data.get("status", "Pending"),
                "role": data.get("role")
            })

        return {"count": len(providers), "service_providers": providers}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))