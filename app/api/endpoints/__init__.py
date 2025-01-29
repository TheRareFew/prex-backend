from fastapi import APIRouter
from .user_message import router as user_message_router
from .article import router as article_router
from .ticket import router as ticket_router
from .document import router as document_router

api_router = APIRouter()

api_router.include_router(user_message_router, tags=["user"])
api_router.include_router(article_router, tags=["articles"])
api_router.include_router(ticket_router, tags=["tickets"])
api_router.include_router(document_router, tags=["documents"]) 