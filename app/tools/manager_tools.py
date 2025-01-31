from typing import Dict, Any
from langchain.tools import tool
from pydantic import UUID4
from datetime import datetime
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_slug(text: str) -> str:
    """Create URL-friendly slug from text."""
    # Convert to lowercase
    slug = text.lower()
    # Replace spaces with hyphens
    slug = re.sub(r'[\s]+', '-', slug)
    # Remove special characters
    slug = re.sub(r'[^\w\-]', '', slug)
    # Remove multiple hyphens
    slug = re.sub(r'\-+', '-', slug)
    # Trim hyphens from ends
    slug = slug.strip('-')
    return slug

@tool
async def write_article(prompt: str) -> Dict[str, Any]:
    """Generate an article based on the manager's prompt. Article content should be formatted in HTML for rich text. Article title and description should not have any HTML tags."""
    try:
        # Generate a title from the prompt (first sentence or up to 100 chars)
        title = prompt.split('.')[0].strip()
        if len(title) > 100:
            title = title[:97] + "..."

        # Generate a slug from the title
        slug = make_slug(title)

        # Current timestamp for created_at and updated_at
        current_time = datetime.utcnow().isoformat()

        # Convert the prompt into HTML-formatted content
        # Note: In real implementation, this would use more sophisticated text processing
        content = f"<div><p>{prompt}</p></div>"

        # Generate a brief description (first 2-3 sentences up to 200 chars)
        description = ". ".join(prompt.split('.')[:2]).strip()
        if len(description) > 200:
            description = description[:197] + "..."

        return {
            "action": "write_article",
            "article": {
                "title": title,
                "description": description,
                "content": content,
                "status": "pending_approval",  # Changed from draft to pending_approval
                "created_at": current_time,
                "updated_at": current_time,
                "published_at": None,  # Will be set when article is approved
                "view_count": 0,
                "is_faq": False,  # Default to regular article
                "category": "general",  # Default category
                "slug": slug
            }
        }
    except Exception as e:
        logger.error(f"Article generation error: {str(e)}")
        raise

@tool
async def update_article_status(article_id: UUID4, new_status: str, reason: str = "") -> Dict[str, Any]:
    """Update the status of an article."""
    try:
        valid_statuses = ["draft", "pending_approval", "approved", "rejected", "archived"]
        if new_status not in valid_statuses:
            raise ValueError(f"Invalid status. Must be one of: {', '.join(valid_statuses)}")

        current_time = datetime.utcnow().isoformat()
        
        # If article is being approved, set published_at
        published_at = current_time if new_status == "approved" else None

        return {
            "action": "update_article_status",
            "article_id": str(article_id),
            "status": new_status,
            "reason": reason,
            "metadata": {
                "updated_at": current_time,
                "published_at": published_at
            }
        }
    except Exception as e:
        logger.error(f"Article status update error: {str(e)}")
        raise

@tool
async def add_article_note(article_id: UUID4, note: str) -> Dict[str, Any]:
    """Add a note to an article."""
    try:
        return {
            "action": "add_article_note",
            "article_id": str(article_id),
            "note": note,
            "metadata": {
                "created_at": datetime.utcnow().isoformat()
            }
        }
    except Exception as e:
        logger.error(f"Add article note error: {str(e)}")
        raise 