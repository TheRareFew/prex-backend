import os
from typing import Dict, Any
from langchain.tools import tool
from pydantic import UUID4
from datetime import datetime
import json
import logging
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from app.core.config import get_settings
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get settings
settings = get_settings()

# Set environment variables
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["PINECONE_API_KEY"] = settings.PINECONE_API_KEY
os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2).lower()
os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

# Initialize embeddings for different dimensions
kb_embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL_1536      # 1536 dimensions for articles
)
file_embeddings = OpenAIEmbeddings(
    model=settings.EMBEDDING_MODEL_3072      # 3072 dimensions for file chunks
)
logger.info("Initialized embeddings")

# Pre-initialize vector stores
kb_vectorstore = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX_TWO,
    embedding=kb_embeddings,
    namespace="articles"
)
chunks_vectorstore = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX,
    embedding=file_embeddings,
    namespace="chunks"
)
descriptions_vectorstore = PineconeVectorStore(
    index_name=settings.PINECONE_INDEX,
    embedding=file_embeddings,
    namespace="descriptions"
)
logger.info("Initialized vector stores")

@tool
async def record_feature_request(message: str) -> Dict[str, Any]:
    """Record a feature request from the user message."""
    try:
        return {
            "action": "feature_request",
            "message": message,
            "metadata": {
                "category": "feature_request",  # matches ticket_category_type
                "priority": "low",              # matches ticket_priority_type
                "status": "fresh"               # matches ticket_status_type
            }
        }
    except Exception as e:
        logger.error(f"Feature request error: {str(e)}")
        raise

@tool
async def record_feedback(message: str) -> Dict[str, Any]:
    """Record feedback from the user message."""
    try:
        return {
            "action": "feedback",
            "message": message,
            "metadata": {
                "category": "feedback",    # matches ticket_category_type
                "priority": "low",         # matches ticket_priority_type
                "status": "fresh"          # matches ticket_status_type
            }
        }
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise

@tool
async def search_kb(query: str) -> Dict[str, Any]:
    """Search knowledge base (approved articles) for relevant information."""
    try:
        # Search for approved articles using pre-initialized store
        articles = kb_vectorstore.similarity_search(
            query,
            k=5,  # Top 5 articles
            filter={
                "status": "approved",  # matches article_status_type
                "is_faq": True        # prioritize FAQs
            }
        )

        # Format results
        results = [{
            "title": doc.metadata.get("title", ""),
            "content": doc.page_content,
            "category": doc.metadata.get("category", "general"),  # matches article_category_type
            "published_at": doc.metadata.get("published_at", ""),
            "slug": doc.metadata.get("slug", "")
        } for doc in articles]

        return {
            "action": "search_kb",
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"KB search error: {str(e)}")
        raise

@tool
async def search_info_store(query: str) -> Dict[str, Any]:
    """Search information store (file chunks and descriptions) for relevant information."""
    try:
        # Search both chunks and descriptions using pre-initialized stores
        chunks = chunks_vectorstore.similarity_search(
            query,
            k=3  # Top 3 chunks
        )
        descriptions = descriptions_vectorstore.similarity_search(
            query,
            k=2  # Top 2 descriptions
        )

        # Format results
        results = {
            "chunks": [{
                "content": doc.page_content,
                "file_name": doc.metadata.get("file_name", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "chunk_index": doc.metadata.get("chunk_index", 0),
                "uploaded_at": doc.metadata.get("uploaded_at", "")
            } for doc in chunks],
            "descriptions": [{
                "content": doc.page_content,
                "file_name": doc.metadata.get("file_name", ""),
                "file_type": doc.metadata.get("file_type", ""),
                "uploaded_at": doc.metadata.get("uploaded_at", "")
            } for doc in descriptions]
        }

        return {
            "action": "search_info",
            "query": query,
            "results": results
        }
    except Exception as e:
        logger.error(f"Info store search error: {str(e)}")
        raise

@tool
async def escalate_ticket(reason: str) -> Dict[str, Any]:
    """Escalate the ticket to human support."""
    try:
        return {
            "action": "escalate",
            "reason": reason,
            "metadata": {
                "priority": "high",     # matches ticket_priority_type
                "status": "fresh",      # matches ticket_status_type
                "category": "general"   # matches ticket_category_type
            }
        }
    except Exception as e:
        logger.error(f"Escalate error: {str(e)}")
        raise 
