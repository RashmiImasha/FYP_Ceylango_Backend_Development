from fastapi import APIRouter
from google.cloud import firestore
from app.models.chatbot import ChatRequest, ChatMessage, ChatSession
from app.database.connection import chatbot_history_collection
from app.services.chatbot_service import MultilingualRAGChatbot
import logging, uuid
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()
chatbot = MultilingualRAGChatbot()

@router.post("/message")
def chat_message(request: ChatRequest):
    
    session_id = request.session_id or str(uuid.uuid4())    
    location = (request.lat, request.lon) if request.lat and request.lon else None

    # Get existing chat history
    chat_history = []
    try:       
        chat_ref = chatbot_history_collection.document(session_id)
        session_doc = chat_ref.get()
        
        if session_doc.exists:
            chat_history = session_doc.to_dict().get('messages', [])
    except Exception as e:
        logger.error(f"Failed to fetch history: {e}")

    response = chatbot.chat(request.query, location, chat_history=chat_history)
    
    # Save chat history
    try:        
        chat_ref = chatbot_history_collection.document(session_id)
        
        # Check if session exists
        session_doc = chat_ref.get()
        
        if session_doc.exists:
            # Append to existing session
            chat_ref.update({
                'messages': firestore.ArrayUnion([
                    {
                        'role': 'user',
                        'content': request.query,
                        'timestamp': datetime.now()
                    },
                    {
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now()
                    }
                ]),
                'updated_at': datetime.now()
            })
        else:
            # Create new session
            chat_ref.set({
                'session_id': session_id,
                'user_id': request.user_id,
                'messages': [
                    {
                        'role': 'user',
                        'content': request.query,
                        'timestamp': datetime.now()
                    },
                    {
                        'role': 'assistant',
                        'content': response,
                        'timestamp': datetime.now()
                    }
                ],
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
        
        logger.info(f"Chat saved to session: {session_id}")
        
    except Exception as e:
        logger.error(f"Failed to save chat history: {e}")
    
    return {
        "response": response,
        "session_id": session_id  
    }
    

@router.get("/history/{session_id}")
def get_chat_history(session_id: str):
    """Get chat history for a session"""
    try:        
        chat_ref = chatbot_history_collection.document(session_id)
        session_doc = chat_ref.get()
        
        if not session_doc.exists:
            return {"error": "Session not found", "messages": []}
        
        session_data = session_doc.to_dict()
        
        return {
            "session_id": session_id,
            "messages": session_data.get('messages', []),
            "created_at": session_data.get('created_at'),
            "updated_at": session_data.get('updated_at')
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch chat history: {e}")
        return {"error": str(e), "messages": []}


@router.get("/history/user/{user_id}")
def get_user_chat_sessions(user_id: str):
    try:
        # Try the indexed query first
        sessions = chatbot_history_collection\
            .where('user_id', '==', user_id)\
            .order_by('updated_at', direction=firestore.Query.DESCENDING)\
            .limit(50)\
            .stream()
        
        sessions_list = []
        for doc in sessions:
            data = doc.to_dict()
            messages = data.get('messages', [])
            last_message = messages[-1] if messages else {}
            
            sessions_list.append({
                'session_id': doc.id,  # Use document ID
                'last_message': last_message.get('content', '')[:100],
                'updated_at': data.get('updated_at'),
                'message_count': len(messages)
            })
        
        return {"sessions": sessions_list}
        
    except Exception as e:
        if "index" in str(e).lower():
            logger.info("Index not ready, using fallback query")
            # Fallback: Get without ordering first, then sort manually
            try:
                all_sessions = chatbot_history_collection\
                    .where('user_id', '==', user_id)\
                    .limit(50)\
                    .stream()
                
                sessions_list = []
                for doc in all_sessions:
                    data = doc.to_dict()
                    messages = data.get('messages', [])
                    last_message = messages[-1] if messages else {}
                    
                    sessions_list.append({
                        'session_id': doc.id,
                        'last_message': last_message.get('content', '')[:100],
                        'updated_at': data.get('updated_at'),
                        'message_count': len(messages)
                    })
                
                # Manual sorting
                sessions_list.sort(
                    key=lambda x: x.get('updated_at') or datetime.min, 
                    reverse=True
                )
                
                return {"sessions": sessions_list}
                
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return {"error": "Index is building, please try again in a few minutes", "sessions": []}
        else:
            logger.error(f"Failed to fetch user sessions: {e}")
            return {"error": str(e), "sessions": []}
        
@router.delete("/history/{session_id}")
def delete_chat_history(session_id: str):

    try:
        chat_ref = chatbot_history_collection.document(session_id)
        session_doc = chat_ref.get()
        
        if not session_doc.exists:
            return {"error": "Session not found", "success": False}
        
        # Delete the session
        chat_ref.delete()        
        logger.info(f"Chat session {session_id} deleted successfully")
        
        return {
            "message": "Chat history deleted successfully",
            "session_id": session_id,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Failed to delete chat history: {e}")
        return {"error": str(e), "success": False}
