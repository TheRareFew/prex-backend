from fastapi import APIRouter
from app.api.endpoints.user_message import router as user_message_router
from app.api.endpoints.ticket import router as ticket_router
from app.api.endpoints.article import router as article_router
from app.api.endpoints.document import router as document_router
from app.api.endpoints.manager_prompt import router as manager_prompt_router

router = APIRouter()

router.include_router(user_message_router, tags=["user"])
router.include_router(article_router, tags=["articles"])
router.include_router(ticket_router, tags=["tickets"])
router.include_router(document_router, tags=["documents"])
router.include_router(manager_prompt_router, tags=["manager"])

__all__ = [
    "router",
    "user_message_router",
    "ticket_router",
    "article_router",
    "document_router",
    "manager_prompt_router"
] 