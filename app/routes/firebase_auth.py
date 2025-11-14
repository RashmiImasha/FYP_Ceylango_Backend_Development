from fastapi import Header, HTTPException, status
from typing import Optional
from firebase_admin import auth
import logging

logger = logging.getLogger(__name__)

def verify_token(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        logger.warning("Missing or invalid authorization header")
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    id_token = authorization.split(" ")[1]

    try:
        decoded_token = auth.verify_id_token(id_token)
        logger.info(f"Token verified for user: {decoded_token['uid']}")
        return decoded_token
    except Exception:
        logger.error("Token verification failed", exc_info=True)
        raise HTTPException(status_code=401, detail="Invalid or expired token")
