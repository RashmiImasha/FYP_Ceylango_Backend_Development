# app/routes/chat.py

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from firebase_admin import auth, messaging
from app.database.connection import db
from app.models.chat import (
    CreateConversationRequest,
    ConversationResponse,
    ConversationListItem,
    UserStatusUpdate
)
from typing import List
import google.cloud.firestore

router = APIRouter()
security = HTTPBearer()

# Collections
conversations_collection = db.collection("conversations")
users_collection = db.collection("users")

# ==================== Authentication Middleware ====================

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase ID token and return user data"""
    try:
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

# ==================== Conversation Routes ====================

@router.post("/conversations/create", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Create a new conversation between tourist and service provider
    Backend validates both users exist before creating
    """
    try:
        # Verify user is one of the participants
        if current_user['uid'] not in [request.tourist_id, request.provider_id]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only create conversations you're part of"
            )

        # Check if conversation already exists
        existing_query = conversations_collection.where(
            filter=google.cloud.firestore.FieldFilter(
                f"participants.{request.tourist_id}", "==", True
            )
        ).where(
            filter=google.cloud.firestore.FieldFilter(
                f"participants.{request.provider_id}", "==", True
            )
        ).limit(1)

        existing = list(existing_query.stream())

        if existing:
            conversation_data = existing[0].to_dict()
            participant_details = conversation_data.get("participantDetails", {})
            
            return ConversationResponse(
                conversation_id=existing[0].id,
                exists=True,
                tourist_name=participant_details.get(request.tourist_id, {}).get("name"),
                provider_name=participant_details.get(request.provider_id, {}).get("name")
            )

        # Get user details from Firestore
        tourist_doc = users_collection.document(request.tourist_id).get()
        provider_doc = users_collection.document(request.provider_id).get()

        if not tourist_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Tourist not found"
            )
        
        if not provider_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Service provider not found"
            )

        tourist_data = tourist_doc.to_dict()
        provider_data = provider_doc.to_dict()

        # Verify provider is approved (if they have status field)
        if provider_data.get("status") == "Pending":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Service provider application is still pending"
            )

        # Create new conversation document
        conversation_ref = conversations_collection.document()
        conversation_data = {
            "participants": {
                request.tourist_id: True,
                request.provider_id: True
            },
            "participantDetails": {
                request.tourist_id: {
                    "name": tourist_data.get("full_name", "Tourist"),
                    "avatar": tourist_data.get("avatar", ""),
                    "role": tourist_data.get("role", "tourist")
                },
                request.provider_id: {
                    "name": provider_data.get("full_name", "Provider"),
                    "avatar": provider_data.get("avatar", ""),
                    "role": provider_data.get("role", "service_provider")
                }
            },
            "lastMessage": "",
            "lastMessageSender": "",
            "lastMessageTime": google.cloud.firestore.SERVER_TIMESTAMP,
            "unreadCount": {
                request.tourist_id: 0,
                request.provider_id: 0
            },
            "createdAt": google.cloud.firestore.SERVER_TIMESTAMP
        }

        conversation_ref.set(conversation_data)

        return ConversationResponse(
            conversation_id=conversation_ref.id,
            exists=False,
            tourist_name=tourist_data.get("full_name"),
            provider_name=provider_data.get("full_name")
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating conversation: {str(e)}"
        )


@router.get("/conversations", response_model=List[ConversationListItem])
async def get_user_conversations(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user['uid']
        
        # Alternative query approach
        all_conversations = conversations_collection.stream()
        
        conversations = []
        for doc in all_conversations:
            data = doc.to_dict()
            participants = data.get("participants", {})
            
            # Check if current user is a participant
            if user_id in participants:
                # Get other participant
                other_user_id = None
                for pid in participants.keys():
                    if pid != user_id:
                        other_user_id = pid
                        break
                
                if other_user_id:
                    participant_details = data.get("participantDetails", {})
                    other_user = participant_details.get(other_user_id, {})
                    
                    # Handle timestamp safely
                    last_message_time = data.get("lastMessageTime")
                    formatted_time = None
                    if hasattr(last_message_time, 'isoformat'):
                        formatted_time = last_message_time.isoformat()
                    elif last_message_time:
                        formatted_time = str(last_message_time)
                    
                    conversations.append(ConversationListItem(
                        conversation_id=doc.id,
                        other_user_id=other_user_id,
                        other_user_name=other_user.get("name", "Unknown User"),
                        other_user_avatar=other_user.get("avatar", ""),
                        other_user_role=other_user.get("role", "user"),
                        last_message=data.get("lastMessage", "No messages yet"),
                        last_message_time=formatted_time,
                        unread_count=data.get("unreadCount", {}).get(user_id, 0),
                        is_last_message_mine=data.get("lastMessageSender") == user_id
                    ))
        
        # Sort by last message time manually
        conversations.sort(key=lambda x: x.last_message_time or "", reverse=True)
        
        return conversations
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/conversations/{conversation_id}")
async def get_conversation_details(
    conversation_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get details of a specific conversation
    Verifies user is a participant
    """
    try:
        user_id = current_user['uid']
        
        conversation_ref = conversations_collection.document(conversation_id)
        conversation_doc = conversation_ref.get()

        if not conversation_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found"
            )

        conversation_data = conversation_doc.to_dict()
        participants = conversation_data.get("participants", {})

        if user_id not in participants:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not a participant in this conversation"
            )

        return {
            "conversation_id": conversation_id,
            **conversation_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching conversation: {str(e)}"
        )


# ==================== User Status Routes ====================

@router.put("/user/status")
async def update_user_status(
    status_update: UserStatusUpdate,
    current_user: dict = Depends(get_current_user)
):
    """
    Update user online/offline status and FCM token
    Called when user opens/closes app
    """
    try:
        user_id = current_user['uid']
        user_ref = users_collection.document(user_id)

        update_data = {
            "online": status_update.online,
            "lastSeen": google.cloud.firestore.SERVER_TIMESTAMP
        }

        if status_update.fcm_token:
            update_data["fcmToken"] = status_update.fcm_token

        user_ref.update(update_data)

        return {
            "success": True, 
            "online": status_update.online,
            "message": f"Status updated to {'online' if status_update.online else 'offline'}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating status: {str(e)}"
        )


# ==================== Health Check ====================

@router.get("/health")
async def chat_health_check():
    """Check if chat service is running"""
    return {
        "status": "healthy",
        "service": "chat",
        "message": "Chat service is running"
    }